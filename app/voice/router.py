"""
Voice Assistant Router for LiveKit.
Provides endpoints for token generation and session management.
"""

import uuid
import logging
from fastapi import APIRouter, Depends, HTTPException
from typing import Optional

from app.models.database import User
from app.models.schemas import VoiceTokenRequest, VoiceTokenResponse
from app.auth.dependencies import get_current_user, require_permission
from app.voice.livekit_agent import livekit_manager
from app.config import get_settings

router = APIRouter(prefix="/voice", tags=["Voice"])
logger = logging.getLogger(__name__)
settings = get_settings()

@router.get("/status")
async def voice_status():
    """Check if voice features are available."""
    return {
        "available": livekit_manager.is_available,
        "message": "Voice ready" if livekit_manager.is_available else "LiveKit not configured.",
    }

@router.post("/token", response_model=VoiceTokenResponse)
async def get_voice_token(
    request: VoiceTokenRequest,
    current_user: User = Depends(require_permission("use_voice")),
):
    """
    Generate a LiveKit token to join a voice call room.
    """
    if not livekit_manager.is_available:
        raise HTTPException(
            status_code=503,
            detail="Voice features not available. LiveKit server is not configured.",
        )

    # Use existing session_id if provided, otherwise create new
    session_id = request.session_id or str(uuid.uuid4())
    # We prefix with voice_ and tenant_id to keep rooms isolated
    room_name = f"voice_{session_id}"
    
    # Create the room via LiveKit API
    room_result = await livekit_manager.create_room(
        room_name=room_name,
        tenant_id=current_user.tenant_id,
    )

    if not room_result.get("created") and room_result.get("error"):
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create voice room: {room_result['error']}",
        )

    dispatch_result = await livekit_manager.dispatch_agent(
        room_name=room_name,
        tenant_id=current_user.tenant_id,
    )
    if not dispatch_result.get("created"):
        raise HTTPException(
            status_code=500,
            detail=f"Failed to dispatch voice agent: {dispatch_result['error']}",
        )

    # Generate participant token
    # We pass the metadata so the worker can identify the tenant/user
    token = await livekit_manager.create_voice_token(
        room_name=room_name,
        participant_name=current_user.username,
        tenant_id=current_user.tenant_id,
    )

    if token is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate voice token",
        )

    return VoiceTokenResponse(
        token=token,
        url=settings.livekit_url,
        room_name=f"{current_user.tenant_id}_{room_name}",
        session_id=session_id,
    )

@router.get("/rooms")
async def list_voice_rooms(
    current_user: User = Depends(require_permission("use_voice")),
):
    """List active voice rooms for the current user's tenant."""
    if not livekit_manager.is_available:
        return {"rooms": []}

    rooms = await livekit_manager.list_rooms(tenant_id=current_user.tenant_id)
    return {"rooms": rooms}

@router.delete("/rooms/{room_name}")
async def end_voice_session(
    room_name: str,
    current_user: User = Depends(require_permission("use_voice")),
):
    """End a voice session by deleting the room."""
    if not livekit_manager.is_available:
        raise HTTPException(status_code=503, detail="Voice not available")

    success = await livekit_manager.delete_room(
        room_name=room_name,
        tenant_id=current_user.tenant_id,
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to end voice session")

    return {"message": "Voice session ended", "room": room_name}
