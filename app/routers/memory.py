"""
Memory management API routes.
Allows users to view, search, and manage their cross-chat memories.
"""

from fastapi import APIRouter, Depends, HTTPException
from app.models.database import User
from app.models.schemas import MemorySearchRequest, MemorySearchResponse, MemoryEntry
from app.auth.dependencies import get_current_user, require_permission
from app.memory.mem0_client import memory_manager

router = APIRouter(prefix="/memory", tags=["Memory"])


@router.post("/search", response_model=MemorySearchResponse)
async def search_memories(
    request: MemorySearchRequest,
    current_user: User = Depends(require_permission("view_memory")),
):
    """
    Search through the user's cross-chat memories.
    Returns the most relevant memories based on semantic similarity.
    """
    results = await memory_manager.search_memory(
        query=request.query,
        user_id=current_user.id,
        tenant_id=current_user.tenant_id,
        limit=request.limit,
    )

    return MemorySearchResponse(
        results=[
            MemoryEntry(
                id=r["id"],
                content=r["content"],
                metadata=r.get("metadata"),
                created_at=r.get("created_at"),
            )
            for r in results
        ]
    )


@router.get("/", response_model=list[MemoryEntry])
async def list_all_memories(
    current_user: User = Depends(require_permission("view_memory")),
):
    """Get all memories for the current user."""
    results = await memory_manager.get_all_memories(
        user_id=current_user.id,
        tenant_id=current_user.tenant_id,
    )

    return [
        MemoryEntry(
            id=r["id"],
            content=r["content"],
            metadata=r.get("metadata"),
            created_at=r.get("created_at"),
        )
        for r in results
    ]


@router.delete("/{memory_id}")
async def delete_memory(
    memory_id: str,
    current_user: User = Depends(require_permission("manage_memory")),
):
    """Delete a specific memory entry."""
    success = await memory_manager.delete_memory(memory_id)
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found or could not be deleted")
    return {"message": "Memory deleted", "id": memory_id}


@router.delete("/")
async def clear_all_memories(
    current_user: User = Depends(require_permission("manage_memory")),
):
    """Delete all memories for the current user."""
    success = await memory_manager.delete_user_memories(
        user_id=current_user.id,
        tenant_id=current_user.tenant_id,
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to clear memories")
    return {"message": "All memories cleared"}
