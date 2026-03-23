"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Literal
from datetime import datetime


# ──────────────────────────────────────────────
# Auth Schemas
# ──────────────────────────────────────────────

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: str
    tenant_id: str
    role: str


class UserRegister(BaseModel):
    email: str = Field(..., description="User email address")
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=8)
    tenant_slug: str = Field(..., description="Tenant identifier")


class UserLogin(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    role: str
    tenant_id: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ──────────────────────────────────────────────
# Tenant Schemas
# ──────────────────────────────────────────────

class TenantCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=255)
    slug: str = Field(..., min_length=3, max_length=100, pattern="^[a-z0-9-]+$")
    max_users: int = Field(default=50, ge=1)


class TenantResponse(BaseModel):
    id: str
    name: str
    slug: str
    is_active: bool
    max_users: int
    created_at: datetime

    class Config:
        from_attributes = True


# ──────────────────────────────────────────────
# Chat Schemas
# ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = Field(None, description="Existing session ID, or None to create new")


class ChatMessageResponse(BaseModel):
    id: str
    role: str
    content: str
    agent_name: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ChatResponse(BaseModel):
    session_id: str
    response: str
    agent_used: Optional[str] = None
    sources: Optional[List[str]] = None
    memory_updated: bool = False
    agent_trace: Optional[List[str]] = None


class ChatSessionResponse(BaseModel):
    id: str
    title: str
    is_active: bool
    created_at: datetime
    message_count: int = 0

    class Config:
        from_attributes = True


class StreamingChatResponse(BaseModel):
    """For SSE streaming responses."""
    event: Literal["token", "agent_switch", "source", "done", "error"]
    data: str


# ──────────────────────────────────────────────
# Memory Schemas
# ──────────────────────────────────────────────

class MemoryEntry(BaseModel):
    id: str
    content: str
    metadata: Optional[dict] = None
    created_at: Optional[str] = None


class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=50)


class MemorySearchResponse(BaseModel):
    results: List[MemoryEntry]


# ──────────────────────────────────────────────
# Voice Schemas
# ──────────────────────────────────────────────

class VoiceTokenRequest(BaseModel):
    session_id: Optional[str] = None


class VoiceTokenResponse(BaseModel):
    token: str
    url: str
    room_name: str
    session_id: str
