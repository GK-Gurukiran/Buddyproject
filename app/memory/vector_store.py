"""
Qdrant vector store integration for multi-tenant memory storage.
Provides tenant-isolated vector search capabilities.

Falls back to simple in-memory storage when Qdrant is not available.
"""

import uuid
import logging
from typing import List, Optional, Dict, Any
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# In-memory fallback store
# ──────────────────────────────────────────────

class InMemoryVectorStore:
    """Simple in-memory fallback when Qdrant is not available."""

    def __init__(self):
        self._entries: List[Dict[str, Any]] = []

    async def ensure_collection(self):
        pass  # no-op

    async def store_memory(
        self,
        content: str,
        tenant_id: str,
        user_id: str,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        point_id = str(uuid.uuid4())
        entry = {
            "id": point_id,
            "content": content,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "session_id": session_id,
            **(metadata or {}),
        }
        self._entries.append(entry)
        # Keep max 500 entries
        if len(self._entries) > 500:
            self._entries = self._entries[-500:]
        return point_id

    async def search_memories(
        self,
        query: str,
        tenant_id: str,
        user_id: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        # Filter by tenant
        filtered = [e for e in self._entries if e.get("tenant_id") == tenant_id]
        if user_id:
            filtered = [e for e in filtered if e.get("user_id") == user_id]

        # Simple keyword matching
        query_words = query.lower().split()
        scored = []
        for entry in filtered:
            content = entry.get("content", "").lower()
            score = sum(1 for w in query_words if w in content)
            if score > 0:
                scored.append({
                    "id": entry["id"],
                    "content": entry["content"],
                    "score": score / max(len(query_words), 1),
                    "metadata": {k: v for k, v in entry.items()
                                 if k not in ("id", "content", "tenant_id", "user_id")},
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    async def delete_tenant_memories(self, tenant_id: str):
        self._entries = [e for e in self._entries if e.get("tenant_id") != tenant_id]


# ──────────────────────────────────────────────
# Qdrant-backed vector store
# ──────────────────────────────────────────────

class QdrantVectorStore:
    """Qdrant vector store with multi-tenant data isolation."""

    def __init__(self):
        from qdrant_client import QdrantClient
        from langchain_openai import OpenAIEmbeddings

        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key or None,
        )

        # Use OpenRouter or OpenAI for embeddings
        if settings.openrouter_api_key:
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
            )
        else:
            self.embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)

        self.collection_name = settings.qdrant_collection_name

    async def ensure_collection(self):
        """Create the collection if it doesn't exist."""
        from qdrant_client.models import Distance, VectorParams

        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1536,
                    distance=Distance.COSINE,
                ),
            )

    async def store_memory(
        self,
        content: str,
        tenant_id: str,
        user_id: str,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a memory entry with tenant isolation."""
        from qdrant_client.models import PointStruct

        await self.ensure_collection()

        embedding = await self.embeddings.aembed_query(content)
        point_id = str(uuid.uuid4())

        payload = {
            "content": content,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "session_id": session_id,
            **(metadata or {}),
        }

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )

        return point_id

    async def search_memories(
        self,
        query: str,
        tenant_id: str,
        user_id: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for relevant memories with tenant-scoped filtering."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        await self.ensure_collection()

        query_embedding = await self.embeddings.aembed_query(query)

        must_conditions = [
            FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))
        ]
        if user_id:
            must_conditions.append(
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=Filter(must=must_conditions),
            limit=limit,
        )

        return [
            {
                "id": str(hit.id),
                "content": hit.payload.get("content", ""),
                "score": hit.score,
                "metadata": {
                    k: v for k, v in hit.payload.items()
                    if k not in ("content", "tenant_id", "user_id")
                },
            }
            for hit in results
        ]

    async def delete_tenant_memories(self, tenant_id: str):
        """Delete all memories for a specific tenant."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
            ),
        )


# ──────────────────────────────────────────────
# Factory: select best available backend
# ──────────────────────────────────────────────

class VectorStoreManager:
    """
    Manages vector store with automatic backend selection.
    Uses Qdrant when available, falls back to in-memory store.
    """

    def __init__(self):
        self._store = None
        self._using_fallback = False

    def _get_store(self):
        if self._store is not None:
            return self._store

        try:
            self._store = QdrantVectorStore()
            # Test connection
            self._store.client.get_collections()
            logger.info("Qdrant vector store connected successfully")
        except Exception as e:
            logger.warning(f"Qdrant not available ({e}), using in-memory vector store fallback")
            self._store = InMemoryVectorStore()
            self._using_fallback = True

        return self._store

    @property
    def is_using_fallback(self) -> bool:
        self._get_store()
        return self._using_fallback

    async def ensure_collection(self):
        store = self._get_store()
        await store.ensure_collection()

    async def store_memory(self, content, tenant_id, user_id, session_id, metadata=None):
        store = self._get_store()
        return await store.store_memory(content, tenant_id, user_id, session_id, metadata)

    async def search_memories(self, query, tenant_id, user_id=None, limit=5):
        store = self._get_store()
        return await store.search_memories(query, tenant_id, user_id, limit)

    async def delete_tenant_memories(self, tenant_id):
        store = self._get_store()
        return await store.delete_tenant_memories(tenant_id)


# Singleton instance
vector_store = VectorStoreManager()
