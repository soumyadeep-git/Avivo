"""
Core RAG Engine integrating ChromaDB for semantic search/caching,
and Groq API for ultra-fast language model generation and multi-modal vision.
"""

import base64
from typing import Dict, Any, List
from collections import defaultdict

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
    TOP_K_RETRIEVAL
)

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
        if len(self.user_history[user_id]) > 6:
            self.user_history[user_id] = self.user_history[user_id][-6:]

    def _get_history(self, user_id: int) -> List[Dict[str, str]]:
        """Retrieves user conversation history."""
        return self.user_history[user_id]

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
        
        # --- 1. Semantic Caching Check (Efficiency) ---
        if self.cache_collection.count() > 0:
            try:
                cache_results = self.cache_collection.query(
                    query_texts=[query_text],
                    n_results=1
                )
                if cache_results["distances"] and len(cache_results["distances"][0]) > 0:
                    distance = cache_results["distances"][0][0]
                    if distance < SIMILARITY_THRESHOLD:
                        cached_answer = cache_results["metadatas"][0][0]["answer"]
                        sources_str = cache_results["metadatas"][0][0]["sources"]
                        sources = sources_str.split(",") if sources_str else []
                        
                        self._add_to_history(user_id, "user", query_text)
                        self._add_to_history(user_id, "assistant", cached_answer)
                        return {"answer": cached_answer, "sources": sources, "cached": True}
            except Exception as e:
                print(f"Cache error: {e}")

        # --- 2. Knowledge Base Retrieval ---
        try:
            kb_results = self.kb_collection.query(
                query_texts=[query_text],
                n_results=TOP_K_RETRIEVAL
            )
        except Exception:
            return {"answer": "The Knowledge Base is empty or unavailable. Please run ingest.py.", "sources": [], "cached": False}
        
        context_chunks = []
        sources = set()
        
        if kb_results["documents"] and len(kb_results["documents"][0]) > 0:
            for idx, doc in enumerate(kb_results["documents"][0]):
                context_chunks.append(doc)
                source_meta = kb_results["metadatas"][0][idx].get("source", "Unknown")
                sources.add(source_meta)
                
        context_str = "\n\n".join(context_chunks) if context_chunks else "No specific relevant context found."
        sources_list = list(sources)
        
        # --- 3. Prompt Construction (Innovation: Clever Prompting) ---
        system_prompt = (
            "You are a helpful, professional AI assistant with a knowledge base. "
            "Use the provided context to answer the user's question accurately. "
            "If the answer isn't in the context, clearly state that, but provide a general helpful answer if possible. "
            "Format your response beautifully using Markdown (bolding, lists) for readability."
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self._get_history(user_id))
        
        prompt_with_context = f"Context from Knowledge Base:\n{context_str}\n\nUser Query: {query_text}"
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
        self.cache_collection.add(
            documents=[query_text],
            metadatas=[{"answer": answer, "sources": ",".join(sources_list)}],
            ids=[f"cache_{hash(query_text)}_{user_id}"]
        )
        
        self._add_to_history(user_id, "user", query_text)
        self._add_to_history(user_id, "assistant", answer)
        
        return {"answer": answer, "sources": sources_list, "cached": False}

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
