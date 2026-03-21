"""
Core RAG Engine integrating the configured vector store,
Groq generation, and retrieval-aware response building.
"""

import base64
from collections import defaultdict
import logging
from typing import Any, Dict, List
import uuid

from groq import AsyncGroq

from config import settings
from logging_utils import log_event
from vector_store import RetrievedChunk, get_vector_store


logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        """Initializes the LLM client, Vector DB, and Memory Buffer."""
        self.groq_client = AsyncGroq(api_key=settings.groq_api_key)
        self.vector_store = get_vector_store()
        self.user_history = defaultdict(list)

    def _add_to_history(self, user_id: int, role: str, content: str) -> None:
        """Maintains the conversation history (last 3 interactions)."""
        self.user_history[user_id].append({"role": role, "content": content})
        if len(self.user_history[user_id]) > settings.max_history_messages:
            self.user_history[user_id] = self.user_history[user_id][-settings.max_history_messages:]

    def _get_history(self, user_id: int) -> List[Dict[str, str]]:
        """Retrieves user conversation history."""
        return self.user_history[user_id]

    @staticmethod
    def _normalize_query(query_text: str) -> str:
        """Normalize user queries so cache hits are more consistent."""
        return " ".join(query_text.strip().lower().split())

    @staticmethod
    def _make_cache_id(user_id: int, query_text: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"cache:{user_id}:{query_text}"))

    @staticmethod
    def _format_source_label(metadata: Dict[str, Any]) -> str:
        source = metadata.get("path") or metadata.get("source", "Unknown")
        section = metadata.get("section")
        if section and section not in {"Introduction", "Untitled Section"}:
            return f"{source} -> {section}"
        return source

    def _build_retrieval_context(self, retrieved_items: List[RetrievedChunk]) -> Dict[str, Any]:
        """Convert retrieval results into prompt context and user-facing citations."""
        relevant_items = [
            item for item in retrieved_items if item.score >= settings.min_relevance_score
        ]
        chosen_items = relevant_items or retrieved_items[:1]

        context_parts: List[str] = []
        source_labels: List[str] = []
        source_snippets: List[str] = []
        total_chars = 0

        for item in chosen_items:
            label = self._format_source_label(item.metadata)
            source_labels.append(label)

            snippet_preview = " ".join(item.content.split())
            if snippet_preview:
                source_snippets.append(f"{label}: {snippet_preview[:180]}")

            context_piece = f"[Source: {label} | similarity={item.score:.3f}]\n{item.content}"
            if total_chars + len(context_piece) > settings.max_context_chars:
                remaining = settings.max_context_chars - total_chars
                if remaining <= 0:
                    break
                context_piece = context_piece[:remaining]

            context_parts.append(context_piece)
            total_chars += len(context_piece)
            if total_chars >= settings.max_context_chars:
                break

        return {
            "context": "\n\n".join(context_parts).strip(),
            "sources": list(dict.fromkeys(source_labels)),
            "source_snippets": source_snippets[:3],
            "has_relevant_context": bool(relevant_items),
        }

    async def query(self, user_id: int, query_text: str) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline with Semantic Caching.
        
        Data Flow:
        1. Query -> Embedding
        2. Embedding -> Check Cache -> Return if Hit
        3. Miss -> Search KB -> Retrieve Context
        4. Context + Query + History -> Groq LLM -> Response
        5. Store in Cache & History -> Return Response
        """
        normalized_query = self._normalize_query(query_text)

        # --- 1. Semantic Caching Check (Efficiency) ---
        try:
            cache_entry = self.vector_store.query_cache(normalized_query)
            if cache_entry:
                log_event(
                    logger,
                    logging.INFO,
                    "semantic_cache_hit",
                    user_id=user_id,
                    cache_score=round(cache_entry.score, 4),
                )
                self._add_to_history(user_id, "user", query_text)
                self._add_to_history(user_id, "assistant", cache_entry.answer)
                return {
                    "answer": cache_entry.answer,
                    "sources": cache_entry.sources,
                    "source_snippets": cache_entry.source_snippets,
                    "cached": True,
                    "grounded": True,
                    "cache_score": cache_entry.score,
                }
        except Exception as exc:
            logger.warning("Cache lookup failed: %s", exc)

        # --- 2. Knowledge Base Retrieval ---
        try:
            retrieved_chunks = self.vector_store.query_knowledge_base(
                query_text=query_text,
                limit=settings.top_k_retrieval,
            )
        except Exception as exc:
            logger.exception("Knowledge base query failed: %s", exc)
            return {
                "answer": "The knowledge base is empty or unavailable. Please run `python ingest.py` first.",
                "sources": [],
                "source_snippets": [],
                "cached": False,
                "grounded": False,
            }

        retrieval_context = self._build_retrieval_context(retrieved_chunks)
        context_str = retrieval_context["context"] or "No relevant context found."
        sources_list = retrieval_context["sources"]
        log_event(
            logger,
            logging.INFO,
            "knowledge_base_retrieved",
            user_id=user_id,
            retrieved_count=len(retrieved_chunks),
            grounded=retrieval_context["has_relevant_context"],
            top_score=round(retrieved_chunks[0].score, 4) if retrieved_chunks else None,
        )

        # --- 3. Prompt Construction (Innovation: Clever Prompting) ---
        system_prompt = (
            "You are a careful RAG assistant. "
            "Answer using the retrieved context whenever possible. "
            "If the context is weak or missing, explicitly say that the answer is based on limited retrieved evidence. "
            "Do not invent citations or claim certainty you do not have. "
            "Prefer concise, directly useful answers written in plain text."
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self._get_history(user_id))

        prompt_with_context = (
            f"Retrieved Context:\n{context_str}\n\n"
            f"Question: {query_text}\n\n"
            "Answer in this structure:\n"
            "1. Direct answer\n"
            "2. Short supporting explanation\n"
            "3. Mention if the retrieved context was limited when applicable\n"
            "Do not use markdown tables. Keep the answer readable in a chat message."
        )
        messages.append({"role": "user", "content": prompt_with_context})

        # --- 4. LLM Generation ---
        try:
            response = await self.groq_client.chat.completions.create(
                messages=messages,
                model=settings.groq_text_model,
                temperature=0.3,
                max_tokens=512,
            )
            answer = response.choices[0].message.content
        except Exception:
            logger.exception("Groq text completion failed")
            return {
                "answer": "The language model is temporarily unavailable. Please try again in a moment.",
                "sources": sources_list,
                "source_snippets": retrieval_context["source_snippets"],
                "cached": False,
                "grounded": retrieval_context["has_relevant_context"],
            }
        
        # --- 5. Update State & Return ---
        cache_id = self._make_cache_id(user_id, normalized_query)
        try:
            self.vector_store.upsert_cache_entry(
                cache_id=cache_id,
                normalized_query=normalized_query,
                answer=answer,
                sources=sources_list,
                source_snippets=retrieval_context["source_snippets"],
            )
        except Exception as exc:
            logger.warning("Failed to upsert semantic cache entry: %s", exc)

        self._add_to_history(user_id, "user", query_text)
        self._add_to_history(user_id, "assistant", answer)
        log_event(
            logger,
            logging.INFO,
            "rag_query_completed",
            user_id=user_id,
            grounded=retrieval_context["has_relevant_context"],
            sources=len(sources_list),
        )

        return {
            "answer": answer,
            "sources": sources_list,
            "source_snippets": retrieval_context["source_snippets"],
            "cached": False,
            "grounded": retrieval_context["has_relevant_context"],
        }

    async def summarize(self, user_id: int) -> str:
        """Summarizes the recent conversation history for a user."""
        history = self._get_history(user_id)
        if not history:
            return "You have no recent conversation history to summarize."
            
        messages = [{"role": "system", "content": "You are an AI that provides concise 1-sentence summaries of conversations."}]
        history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
        messages.append({"role": "user", "content": f"Summarize this:\n{history_text}"})
        
        try:
            response = await self.groq_client.chat.completions.create(
                messages=messages,
                model=settings.groq_text_model,
                temperature=0.5,
                max_tokens=256,
            )
            return response.choices[0].message.content
        except Exception:
            logger.exception("Groq summarization failed")
            return "Summary is temporarily unavailable."

    async def describe_image(self, user_id: int, image_bytes: bytes) -> str:
        """
        Innovation Bonus: Multi-modal Vision Support.
        Describes an image using Groq's fast Vision Model (llama-3.2-11b-vision-preview).
        Generates a short caption and 3 tags.
        """
        # Convert bytes to base64 data URI format for the API
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{base64_image}"

        prompt = (
            "Analyze this image. Provide a short, descriptive caption (1-2 sentences), "
            "followed by exactly 3 relevant keywords or tags. "
            "Format the response nicely:\n"
            "**Caption:** <text>\n"
            "**Tags:** #<tag1> #<tag2> #<tag3>"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]

        try:
            response = await self.groq_client.chat.completions.create(
                messages=messages,
                model=settings.groq_vision_model,
                temperature=0.4,
                max_tokens=256,
            )
            answer = response.choices[0].message.content
        except Exception:
            logger.exception("Groq vision completion failed")
            return "Image analysis is temporarily unavailable."
        
        # Add to history so text bot remembers we discussed an image
        self._add_to_history(user_id, "user", "[Sent an image]")
        self._add_to_history(user_id, "assistant", answer)
        
        return answer

    def health(self) -> Dict[str, Any]:
        """Expose runtime health metadata for health checks."""
        return {
            "vector_store": self.vector_store.health(),
            "history_users": len(self.user_history),
        }
