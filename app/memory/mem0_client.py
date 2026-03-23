"""
Mem0 integration for cross-chat persistent memory.
Provides long-term memory that persists across different chat sessions,
with multi-tenant data isolation.

Falls back to in-memory storage when Qdrant/Mem0 are not available.
"""

import logging
from typing import List, Optional, Dict, Any
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class InMemoryStore:
    """Simple in-memory fallback when Qdrant/Mem0 aren't available."""

    def __init__(self):
        self._store: Dict[str, List[Dict[str, Any]]] = {}

    def add(self, content: str, user_id: str, metadata: Optional[Dict] = None):
        import uuid
        if user_id not in self._store:
            self._store[user_id] = []
        entry = {
            "id": str(uuid.uuid4()),
            "memory": content,
            "metadata": metadata or {},
        }
        self._store[user_id].append(entry)
        # Keep only last 100 memories per user
        self._store[user_id] = self._store[user_id][-100:]
        return entry

    def search(self, query: str, user_id: str, limit: int = 5):
        entries = self._store.get(user_id, [])
        # Simple keyword matching (no vector search without Qdrant)
        query_lower = query.lower()
        scored = []
        for entry in entries:
            content = entry.get("memory", "").lower()
            # Count matching words as a simple relevance score
            words = query_lower.split()
            score = sum(1 for w in words if w in content)
            if score > 0:
                scored.append({**entry, "score": score})

        scored.sort(key=lambda x: x.get("score", 0), reverse=True)
        return scored[:limit]

    def get_all(self, user_id: str):
        return self._store.get(user_id, [])

    def delete(self, memory_id: str):
        for user_entries in self._store.values():
            for i, entry in enumerate(user_entries):
                if entry.get("id") == memory_id:
                    user_entries.pop(i)
                    return True
        return False

    def delete_all(self, user_id: str):
        self._store.pop(user_id, None)


class MemoryManager:
    """
    Manages cross-chat memory using Mem0 + Qdrant backend.
    Falls back to in-memory storage when external services aren't available.
    """

    def __init__(self):
        self._memory = None
        self._using_fallback = False

    def _get_memory(self):
        """Lazy initialization of memory backend."""
        if self._memory is not None:
            return self._memory

        # Try to initialize Mem0 with Qdrant
        try:
            from mem0 import Memory
            config = {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "host": settings.qdrant_host,
                        "port": settings.qdrant_port,
                        "collection_name": f"{settings.qdrant_collection_name}_mem0",
                    },
                },
            }

            # Use OpenRouter or OpenAI for embeddings
            if settings.openrouter_api_key:
                config["llm"] = {
                    "provider": "openai",
                    "config": {
                        "model": settings.openrouter_model,
                        "api_key": settings.openrouter_api_key,
                        "openai_base_url": settings.openrouter_base_url,
                    },
                }
                config["embedder"] = {
                    "provider": "openai",
                    "config": {
                        "model": "text-embedding-3-small",
                        "api_key": settings.openrouter_api_key,
                        "openai_base_url": settings.openrouter_base_url,
                    },
                }
            elif settings.openai_api_key:
                config["llm"] = {
                    "provider": "openai",
                    "config": {
                        "model": settings.openai_model,
                        "api_key": settings.openai_api_key,
                    },
                }
                config["embedder"] = {
                    "provider": "openai",
                    "config": {
                        "model": "text-embedding-3-small",
                        "api_key": settings.openai_api_key,
                    },
                }

            if settings.qdrant_api_key:
                config["vector_store"]["config"]["api_key"] = settings.qdrant_api_key

            self._memory = Memory.from_config(config)
            logger.info("Mem0 + Qdrant memory initialized successfully")

        except Exception as e:
            logger.warning(f"Mem0/Qdrant not available ({e}), using in-memory fallback")
            self._memory = InMemoryStore()
            self._using_fallback = True

        return self._memory

    async def add_memory(
        self,
        content: str,
        user_id: str,
        tenant_id: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store a new memory entry."""
        memory = self._get_memory()
        composite_user_id = f"{tenant_id}:{user_id}"

        extra_metadata = metadata or {}
        extra_metadata["tenant_id"] = tenant_id
        extra_metadata["session_id"] = session_id or ""

        result = memory.add(
            content,
            user_id=composite_user_id,
            metadata=extra_metadata,
        )

        return result

    async def search_memory(
        self,
        query: str,
        user_id: str,
        tenant_id: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search memories relevant to the query."""
        memory = self._get_memory()
        composite_user_id = f"{tenant_id}:{user_id}"

        results = memory.search(
            query=query,
            user_id=composite_user_id,
            limit=limit,
        )

        return [
            {
                "id": r.get("id", ""),
                "content": r.get("memory", r.get("text", "")),
                "score": r.get("score", 0),
                "metadata": r.get("metadata", {}),
            }
            for r in results
        ]

    async def get_all_memories(
        self,
        user_id: str,
        tenant_id: str,
    ) -> List[Dict[str, Any]]:
        """Get all memories for a specific user in a tenant."""
        memory = self._get_memory()
        composite_user_id = f"{tenant_id}:{user_id}"

        results = memory.get_all(user_id=composite_user_id)

        return [
            {
                "id": r.get("id", ""),
                "content": r.get("memory", r.get("text", "")),
                "metadata": r.get("metadata", {}),
                "created_at": r.get("created_at", ""),
            }
            for r in results
        ]

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory entry."""
        try:
            memory = self._get_memory()
            memory.delete(memory_id)
            return True
        except Exception:
            return False

    async def delete_user_memories(self, user_id: str, tenant_id: str) -> bool:
        """Delete all memories for a user within a tenant."""
        try:
            memory = self._get_memory()
            composite_user_id = f"{tenant_id}:{user_id}"
            memory.delete_all(user_id=composite_user_id)
            return True
        except Exception:
            return False


# Singleton instance
memory_manager = MemoryManager()
