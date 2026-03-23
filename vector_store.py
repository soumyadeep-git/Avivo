"""
Vector store abstraction backed by Qdrant for both local development and cloud deployment.
"""

from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cohere
from qdrant_client import QdrantClient, models

from config import settings


logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved vector search result."""

    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    score: float


@dataclass
class CacheEntry:
    """A semantic cache hit."""

    answer: str
    sources: List[str]
    source_snippets: List[str]
    score: float


@dataclass
class IngestionChunk:
    """A chunk ready for vector indexing."""

    chunk_id: str
    content: str
    metadata: Dict[str, Any]


class QdrantVectorStore:
    """Store knowledge-base chunks and semantic cache entries in Qdrant."""

    def __init__(self) -> None:
        self._client = self._create_client()
        self._embedder = cohere.ClientV2(api_key=settings.cohere_api_key)
        self._ensure_collection(settings.knowledge_collection_name)
        self._ensure_collection(settings.cache_collection_name)
        self._ensure_payload_indexes(settings.knowledge_collection_name)

    @staticmethod
    def _create_client() -> QdrantClient:
        if settings.qdrant_url:
            return QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                timeout=settings.request_timeout_seconds,
            )

        Path(settings.qdrant_local_path).mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=settings.qdrant_local_path)

    def _ensure_collection(self, collection_name: str) -> None:
        if self._client.collection_exists(collection_name):
            collection_info = self._client.get_collection(collection_name)
            current_size = collection_info.config.params.vectors.size
            if current_size == settings.embedding_vector_size:
                return
            logger.warning(
                "Recreating collection %s because vector size changed from %s to %s",
                collection_name,
                current_size,
                settings.embedding_vector_size,
            )
            self._client.delete_collection(collection_name)

        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=settings.embedding_vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def _ensure_payload_indexes(self, collection_name: str) -> None:
        """Create payload indexes required for cloud filtering in Qdrant."""
        for field_name in ("path", "document_fingerprint"):
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )

    def _embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        text_list = list(texts)
        if not text_list:
            return []
        response = self._embedder.embed(
            model=settings.embedding_model,
            texts=text_list,
            input_type="search_document",
            embedding_types=["float"],
        )
        return response.embeddings.float_

    def _embed_query(self, query_text: str) -> List[float]:
        response = self._embedder.embed(
            model=settings.embedding_model,
            texts=[query_text],
            input_type="search_query",
            embedding_types=["float"],
        )
        return response.embeddings.float_[0]

    @staticmethod
    def _source_filter(source_path: str) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="path",
                    match=models.MatchValue(value=source_path),
                )
            ]
        )

    def upsert_knowledge_chunks(self, chunks: List[IngestionChunk]) -> int:
        if not chunks:
            return 0

        vectors = self._embed_texts(chunk.content for chunk in chunks)
        points = [
            models.PointStruct(
                id=chunk.chunk_id,
                vector=vectors[index],
                payload={"content": chunk.content, **chunk.metadata},
            )
            for index, chunk in enumerate(chunks)
        ]

        self._client.upsert(
            collection_name=settings.knowledge_collection_name,
            points=points,
            wait=True,
        )
        return len(points)

    def get_document_fingerprint(self, source_path: str) -> str | None:
        records, _ = self._client.scroll(
            collection_name=settings.knowledge_collection_name,
            scroll_filter=self._source_filter(source_path),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None
        return records[0].payload.get("document_fingerprint")

    def delete_document_chunks(self, source_path: str) -> None:
        self._client.delete(
            collection_name=settings.knowledge_collection_name,
            points_selector=models.FilterSelector(filter=self._source_filter(source_path)),
            wait=True,
        )

    def query_knowledge_base(self, query_text: str, limit: int) -> List[RetrievedChunk]:
        response = self._client.query_points(
            collection_name=settings.knowledge_collection_name,
            query=self._embed_query(query_text),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        hits = response.points
        return [
            RetrievedChunk(
                chunk_id=str(hit.id),
                content=str((hit.payload or {}).get("content", "")),
                metadata=dict(hit.payload or {}),
                score=float(hit.score),
            )
            for hit in hits
        ]

    def query_cache(self, normalized_query: str) -> CacheEntry | None:
        response = self._client.query_points(
            collection_name=settings.cache_collection_name,
            query=self._embed_query(normalized_query),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        hits = response.points
        if not hits:
            return None

        top_hit = hits[0]
        if float(top_hit.score) < settings.cache_similarity_threshold:
            return None

        payload = dict(top_hit.payload or {})
        return CacheEntry(
            answer=str(payload.get("answer", "")),
            sources=list(payload.get("sources", [])),
            source_snippets=list(payload.get("source_snippets", [])),
            score=float(top_hit.score),
        )

    def upsert_cache_entry(
        self,
        cache_id: str,
        normalized_query: str,
        answer: str,
        sources: List[str],
        source_snippets: List[str],
    ) -> None:
        vector = self._embed_query(normalized_query)
        point = models.PointStruct(
            id=cache_id,
            vector=vector,
            payload={
                "query": normalized_query,
                "answer": answer,
                "sources": sources,
                "source_snippets": source_snippets,
            },
        )
        self._client.upsert(
            collection_name=settings.cache_collection_name,
            points=[point],
            wait=True,
        )

    def health(self) -> Dict[str, Any]:
        collections = self._client.get_collections()
        return {
            "backend": settings.vector_backend,
            "cloud": settings.use_cloud_vector_store,
            "collections": [item.name for item in collections.collections],
        }


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore()
