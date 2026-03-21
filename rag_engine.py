"""
Core RAG Engine integrating ChromaDB for semantic search/caching,
and Groq API for ultra-fast language model generation and multi-modal vision.
"""

import base64
import hashlib
from typing import Dict, Any, List
from collections import defaultdict
import logging

import chromadb
from chromadb.utils import embedding_functions
from groq import AsyncGroq

from config import (
    DB_DIR,
    COLLECTION_NAME,
    CACHE_COLLECTION_NAME,
    EMBEDDING_MODEL,
    GROQ_TEXT_MODEL,
    GROQ_VISION_MODEL,
    GROQ_API_KEY,
    SIMILARITY_THRESHOLD,
    TOP_K_RETRIEVAL,
    MAX_HISTORY_MESSAGES,
    MAX_CONTEXT_CHARS,
    MIN_RELEVANCE_DISTANCE,
)


logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        """Initializes the LLM client, Vector DB, and Memory Buffer."""
        # LLM Initialization
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        self.groq_client = AsyncGroq(api_key=GROQ_API_KEY)
        
        # Vector DB Initialization (Local SQLite)
        self.chroma_client = chromadb.PersistentClient(path=DB_DIR)
        
        # Local Embedding Function (Efficiency)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        
        # KB Collection
        self.kb_collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_function
        )
        
        # Cache Collection (Efficiency: Semantic Caching)
        self.cache_collection = self.chroma_client.get_or_create_collection(
            name=CACHE_COLLECTION_NAME,
            embedding_function=self.embedding_function
        )
        
        # User Memory Buffer: UserID -> List of Messages (UX: Context awareness)
        # Keeps last 6 messages (3 turns)
        self.user_history = defaultdict(list)

    def _add_to_history(self, user_id: int, role: str, content: str) -> None:
        """Maintains the conversation history (last 3 interactions)."""
        self.user_history[user_id].append({"role": role, "content": content})
        if len(self.user_history[user_id]) > MAX_HISTORY_MESSAGES:
            self.user_history[user_id] = self.user_history[user_id][-MAX_HISTORY_MESSAGES:]

    def _get_history(self, user_id: int) -> List[Dict[str, str]]:
        """Retrieves user conversation history."""
        return self.user_history[user_id]

    @staticmethod
    def _normalize_query(query_text: str) -> str:
        """Normalize user queries so cache hits are more consistent."""
        return " ".join(query_text.strip().lower().split())

    @staticmethod
    def _make_cache_id(user_id: int, query_text: str) -> str:
        digest = hashlib.sha256(f"{user_id}:{query_text}".encode("utf-8")).hexdigest()[:24]
        return f"cache_{digest}"

    @staticmethod
    def _format_source_label(metadata: Dict[str, Any]) -> str:
        source = metadata.get("source", "Unknown")
        section = metadata.get("section")
        if section and section not in {"Introduction", "Untitled Section"}:
            return f"{source} -> {section}"
        return source

    def _build_retrieval_context(self, kb_results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert raw retrieval results into prompt context and user-facing citations."""
        retrieved_items: List[Dict[str, Any]] = []
        documents = kb_results.get("documents") or [[]]
        metadatas = kb_results.get("metadatas") or [[]]
        distances = kb_results.get("distances") or [[]]

        for idx, document in enumerate(documents[0]):
            metadata = metadatas[0][idx] if idx < len(metadatas[0]) else {}
            distance = distances[0][idx] if idx < len(distances[0]) else None
            if document:
                retrieved_items.append(
                    {
                        "content": document,
                        "metadata": metadata,
                        "distance": distance,
                    }
                )

        relevant_items = [
            item
            for item in retrieved_items
            if item["distance"] is None or item["distance"] <= MIN_RELEVANCE_DISTANCE
        ]
        chosen_items = relevant_items or retrieved_items[:1]

        context_parts: List[str] = []
        source_labels: List[str] = []
        source_snippets: List[str] = []
        total_chars = 0

        for item in chosen_items:
            label = self._format_source_label(item["metadata"])
            source_labels.append(label)

            snippet_preview = " ".join(item["content"].split())
            if snippet_preview:
                source_snippets.append(f"{label}: {snippet_preview[:180]}")

            context_piece = f"[Source: {label}]\n{item['content']}"
            if total_chars + len(context_piece) > MAX_CONTEXT_CHARS:
                remaining = MAX_CONTEXT_CHARS - total_chars
                if remaining <= 0:
                    break
                context_piece = context_piece[:remaining]

            context_parts.append(context_piece)
            total_chars += len(context_piece)
            if total_chars >= MAX_CONTEXT_CHARS:
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
        if self.cache_collection.count() > 0:
            try:
                cache_results = self.cache_collection.query(
                    query_texts=[normalized_query],
                    n_results=1
                )
                if cache_results["distances"] and len(cache_results["distances"][0]) > 0:
                    distance = cache_results["distances"][0][0]
                    if distance < SIMILARITY_THRESHOLD:
                        cached_answer = cache_results["metadatas"][0][0]["answer"]
                        sources_str = cache_results["metadatas"][0][0]["sources"]
                        source_snippets_str = cache_results["metadatas"][0][0].get("source_snippets", "")
                        sources = sources_str.split(",") if sources_str else []
                        source_snippets = source_snippets_str.split(" || ") if source_snippets_str else []

                        self._add_to_history(user_id, "user", query_text)
                        self._add_to_history(user_id, "assistant", cached_answer)
                        return {
                            "answer": cached_answer,
                            "sources": sources,
                            "source_snippets": source_snippets,
                            "cached": True,
                            "grounded": True,
                        }
            except Exception as e:
                logger.warning("Cache error: %s", e)

        # --- 2. Knowledge Base Retrieval ---
        try:
            kb_results = self.kb_collection.query(
                query_texts=[query_text],
                n_results=TOP_K_RETRIEVAL,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return {
                "answer": "The knowledge base is empty or unavailable. Please run `python ingest.py` first.",
                "sources": [],
                "source_snippets": [],
                "cached": False,
                "grounded": False,
            }

        retrieval_context = self._build_retrieval_context(kb_results)
        context_str = retrieval_context["context"] or "No relevant context found."
        sources_list = retrieval_context["sources"]

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
        response = await self.groq_client.chat.completions.create(
            messages=messages,
            model=GROQ_TEXT_MODEL,
            temperature=0.3,
            max_tokens=512,
        )
        answer = response.choices[0].message.content
        
        # --- 5. Update State & Return ---
        cache_id = self._make_cache_id(user_id, normalized_query)
        cache_metadata = {
            "answer": answer,
            "sources": ",".join(sources_list),
            "source_snippets": " || ".join(retrieval_context["source_snippets"]),
        }
        try:
            self.cache_collection.upsert(
                documents=[normalized_query],
                metadatas=[cache_metadata],
                ids=[cache_id],
            )
        except Exception as exc:
            logger.warning("Failed to upsert semantic cache entry: %s", exc)

        self._add_to_history(user_id, "user", query_text)
        self._add_to_history(user_id, "assistant", answer)

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
        
        response = await self.groq_client.chat.completions.create(
            messages=messages,
            model=GROQ_TEXT_MODEL,
            temperature=0.5,
            max_tokens=256,
        )
        return response.choices[0].message.content

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

        response = await self.groq_client.chat.completions.create(
            messages=messages,
            model=GROQ_VISION_MODEL,
            temperature=0.4,
            max_tokens=256,
        )
        
        answer = response.choices[0].message.content
        
        # Add to history so text bot remembers we discussed an image
        self._add_to_history(user_id, "user", "[Sent an image]")
        self._add_to_history(user_id, "assistant", answer)
        
        return answer
