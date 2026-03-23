"""
Voice API routes for LiveKit integration.
Handles room creation, token generation, and session management.
Returns clear error messages when LiveKit is not available.
"""

import uuid
from fastapi import APIRouter, Depends, HTTPException

from app.models.database import User
from app.models.schemas import VoiceTokenRequest, VoiceTokenResponse
from app.auth.dependencies import get_current_user, require_permission
from app.voice.livekit_agent import livekit_manager

router = APIRouter(prefix="/voice", tags=["Voice"])


@router.get("/status")
async def voice_status():
    """Check if voice features are available."""
    return {
        "available": livekit_manager.is_available,
        "message": "Voice ready" if livekit_manager.is_available else "LiveKit not configured. Start LiveKit via Docker to enable voice.",
    }


@router.post("/token", response_model=VoiceTokenResponse)
async def get_voice_token(
    request: VoiceTokenRequest,
    current_user: User = Depends(require_permission("use_voice")),
):
    """
    Generate a LiveKit token to join a voice call room.
    Creates a new room if session_id is not provided.
    """
    if not livekit_manager.is_available:
        raise HTTPException(
            status_code=503,
            detail="Voice features not available. LiveKit server is not configured. Run 'docker-compose up livekit' to enable.",
        )

    session_id = request.session_id or str(uuid.uuid4())
    room_name = f"voice_{session_id}"

    # Create the room
    room_result = await livekit_manager.create_room(
        room_name=room_name,
        tenant_id=current_user.tenant_id,
    )

    if not room_result.get("created") and room_result.get("error"):
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create voice room: {room_result['error']}",
        )

    # Generate participant token
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

    from app.config import get_settings
    settings = get_settings()

    return VoiceTokenResponse(
        token=token,
        url=settings.livekit_url or "ws://localhost:7880",
        room_name=f"{current_user.tenant_id}_{room_name}",
        session_id=session_id,
    )


@router.get("/rooms")
async def list_voice_rooms(
    current_user: User = Depends(require_permission("use_voice")),
):
    """List active voice rooms for the current user's tenant."""
    if not livekit_manager.is_available:
        return {"rooms": [], "message": "Voice not available"}

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
