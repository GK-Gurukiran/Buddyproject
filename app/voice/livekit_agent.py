"""
LiveKit voice agent integration.
Handles real-time voice conversations with VAD, STT, TTS pipeline.

Falls back gracefully when LiveKit SDK is not installed or server is unavailable.
"""

import logging
from typing import Optional

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Try to import LiveKit SDK
try:
    from livekit import api as livekit_api
    LIVEKIT_SDK_AVAILABLE = True
except ImportError:
    LIVEKIT_SDK_AVAILABLE = False
    livekit_api = None
    logger.info("LiveKit SDK not installed. Voice features will be unavailable.")


class LiveKitManager:
    """
    Manages LiveKit voice call sessions.
    Handles room creation, token generation, and voice pipeline management.
    Falls back gracefully when LiveKit is not available.
    """

    def __init__(self):
        self._api = None
        self._available = LIVEKIT_SDK_AVAILABLE and bool(settings.livekit_api_key)

    @property
    def is_available(self) -> bool:
        """Check if voice features are available."""
        return self._available

    def _get_api(self):
        """Lazy initialization of LiveKit API client."""
        if not self._available:
            return None

        if self._api is None:
            try:
                self._api = livekit_api.LiveKitAPI(
                    url=settings.livekit_url,
                    api_key=settings.livekit_api_key,
                    api_secret=settings.livekit_api_secret,
                )
            except Exception as e:
                logger.error(f"Failed to initialize LiveKit API: {e}")
                self._available = False
                return None
        return self._api

    async def create_voice_token(
        self,
        room_name: str,
        participant_name: str,
        tenant_id: str,
    ) -> Optional[str]:
        """
        Generate a LiveKit access token for a participant.
        Returns None if LiveKit is not available.
        """
        if not self._available:
            return None

        scoped_room = f"{tenant_id}_{room_name}"

        try:
            token = livekit_api.AccessToken(
                api_key=settings.livekit_api_key,
                api_secret=settings.livekit_api_secret,
            )
            token.with_identity(participant_name)
            token.with_name(participant_name)
            token.with_grants(
                livekit_api.VideoGrants(
                    room_join=True,
                    room=scoped_room,
                )
            )
            return token.to_jwt()
        except Exception as e:
            logger.error(f"Failed to create voice token: {e}")
            return None

    async def create_room(self, room_name: str, tenant_id: str) -> dict:
        """Create a new LiveKit room for a voice session."""
        if not self._available:
            return {"room_name": room_name, "error": "Voice not available (LiveKit not configured)", "created": False}

        scoped_room = f"{tenant_id}_{room_name}"
        lk_api = self._get_api()

        if lk_api is None:
            return {"room_name": scoped_room, "error": "LiveKit API not initialized", "created": False}

        try:
            room = await lk_api.room.create_room(
                livekit_api.CreateRoomRequest(
                    name=scoped_room,
                    empty_timeout=300,
                    max_participants=2,
                )
            )
            return {
                "room_name": room.name,
                "sid": room.sid,
                "created": True,
            }
        except Exception as e:
            logger.error(f"Failed to create room: {e}")
            return {
                "room_name": scoped_room,
                "error": str(e),
                "created": False,
            }

    async def list_rooms(self, tenant_id: str) -> list:
        """List active rooms for a tenant."""
        if not self._available:
            return []

        lk_api = self._get_api()
        if lk_api is None:
            return []

        try:
            rooms = await lk_api.room.list_rooms(livekit_api.ListRoomsRequest())
            prefix = f"{tenant_id}_"
            return [
                {"name": r.name, "participants": r.num_participants, "sid": r.sid}
                for r in rooms
                if r.name.startswith(prefix)
            ]
        except Exception as e:
            logger.error(f"Failed to list rooms: {e}")
            return []

    async def delete_room(self, room_name: str, tenant_id: str) -> bool:
        """Delete a LiveKit room."""
        if not self._available:
            return False

        scoped_room = f"{tenant_id}_{room_name}"
        lk_api = self._get_api()
        if lk_api is None:
            return False

        try:
            await lk_api.room.delete_room(
                livekit_api.DeleteRoomRequest(room=scoped_room)
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete room: {e}")
            return False


# Singleton instance
livekit_manager = LiveKitManager()
