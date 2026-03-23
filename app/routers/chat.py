"""
Chat API routes: send messages, manage sessions, stream responses.
"""

import json
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import User, ChatSession, ChatMessage, get_db
from app.models.schemas import (
    ChatRequest, ChatResponse, ChatSessionResponse, ChatMessageResponse,
)
from app.auth.dependencies import get_current_user, require_permission
from app.auth.rbac import check_tenant_access
from app.agents.graph import agent_graph
from app.memory.mem0_client import memory_manager

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    current_user: User = Depends(require_permission("chat")),
    db: AsyncSession = Depends(get_db),
):
    """
    Send a message to the AI chatbot.
    Creates a new session if session_id is not provided.
    Routes through the multi-agent graph (Supervisor -> Research/Scraper).
    """
    # Get or create session
    session = None
    if request.session_id:
        result = await db.execute(
            select(ChatSession).where(
                ChatSession.id == request.session_id,
                ChatSession.tenant_id == current_user.tenant_id,
            )
        )
        session = result.scalar_one_or_none()
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found",
            )

    if not session:
        session = ChatSession(
            title=request.message[:50] + "..." if len(request.message) > 50 else request.message,
            tenant_id=current_user.tenant_id,
            user_id=current_user.id,
        )
        db.add(session)
        await db.flush()

    # Save user message
    user_msg = ChatMessage(
        session_id=session.id,
        role="user",
        content=request.message,
    )
    db.add(user_msg)
    await db.flush()

    # Retrieve relevant memories for context
    try:
        memories = await memory_manager.search_memory(
            query=request.message,
            user_id=current_user.id,
            tenant_id=current_user.tenant_id,
            limit=5,
        )
        memory_context = "\n".join([m["content"] for m in memories]) if memories else ""
    except Exception:
        memory_context = ""

    # Run through the multi-agent graph
    initial_state = {
        "messages": [],
        "current_agent": "supervisor",
        "user_input": request.message,
        "research_results": None,
        "scrape_results": None,
        "final_response": None,
        "tenant_id": current_user.tenant_id,
        "user_id": current_user.id,
        "session_id": session.id,
        "memory_context": memory_context,
        "sources": [],
        "next_agent": None,
    }

    try:
        result = await agent_graph.ainvoke(initial_state)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent processing error: {str(e)}",
        )

    response_text = result.get("final_response", "I'm sorry, I couldn't generate a response.")
    agent_used = result.get("current_agent", "supervisor")
    sources = result.get("sources", [])
    agent_trace = result.get("agent_trace", None)

    # Save assistant message
    assistant_msg = ChatMessage(
        session_id=session.id,
        role="assistant",
        content=response_text,
        agent_name=agent_used,
        metadata_json=json.dumps({"sources": sources}) if sources else None,
    )
    db.add(assistant_msg)
    await db.flush()

    return ChatResponse(
        session_id=session.id,
        response=response_text,
        agent_used=agent_used,
        sources=sources,
        memory_updated=False,
        agent_trace=agent_trace,
    )


@router.get("/sessions", response_model=list[ChatSessionResponse])
async def list_sessions(
    current_user: User = Depends(require_permission("view_sessions")),
    db: AsyncSession = Depends(get_db),
):
    """List all chat sessions for the current user."""
    result = await db.execute(
        select(ChatSession)
        .where(
            ChatSession.user_id == current_user.id,
            ChatSession.tenant_id == current_user.tenant_id,
        )
        .order_by(ChatSession.updated_at.desc())
    )
    sessions = result.scalars().all()

    response = []
    for s in sessions:
        msg_count = await db.execute(
            select(func.count(ChatMessage.id)).where(ChatMessage.session_id == s.id)
        )
        count = msg_count.scalar() or 0
        response.append(
            ChatSessionResponse(
                id=s.id,
                title=s.title,
                is_active=s.is_active,
                created_at=s.created_at,
                message_count=count,
            )
        )

    return response


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageResponse])
async def get_session_messages(
    session_id: str,
    current_user: User = Depends(require_permission("view_sessions")),
    db: AsyncSession = Depends(get_db),
):
    """Get all messages in a chat session."""
    # Verify session belongs to user's tenant
    session_result = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id)
    )
    session = session_result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not check_tenant_access(current_user.tenant_id, session.tenant_id, current_user.role.value):
        raise HTTPException(status_code=403, detail="Access denied")

    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
    )

    return list(result.scalars().all())


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: User = Depends(require_permission("delete_sessions")),
    db: AsyncSession = Depends(get_db),
):
    """Soft-delete a chat session."""
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.tenant_id == current_user.tenant_id,
        )
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.is_active = False
    await db.flush()

    return {"message": "Session deactivated", "session_id": session_id}
