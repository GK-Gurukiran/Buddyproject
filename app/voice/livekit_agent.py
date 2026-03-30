"""
LiveKit voice agent integration.
"""

import logging
from typing import Optional

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

try:
    from livekit import api as livekit_api
    LIVEKIT_SDK_AVAILABLE = True
except ImportError:
    LIVEKIT_SDK_AVAILABLE = False
    livekit_api = None
    logger.info("LiveKit SDK not installed. Voice features will be unavailable.")


class LiveKitManager:

    def __init__(self):
        self._available = LIVEKIT_SDK_AVAILABLE and bool(settings.livekit_api_key)

    @property
    def is_available(self) -> bool:
        return self._available

    def _make_api(self):
        if not self._available:
            return None
        return livekit_api.LiveKitAPI(
            url=settings.livekit_url,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
        )

    def _scoped(self, tenant_id: str, room_name: str) -> str:
        return f"{tenant_id}_{room_name}"

    async def create_voice_token(
        self,
        room_name: str,
        participant_name: str,
        tenant_id: str,
        participant_identity: str = None,
    ) -> Optional[str]:
        if not self._available:
            return None

        scoped_room = self._scoped(tenant_id, room_name)
        try:
            token = livekit_api.AccessToken(
                api_key=settings.livekit_api_key,
                api_secret=settings.livekit_api_secret,
            )
            token.with_identity(participant_identity or participant_name)
            token.with_name(participant_name)
            token.with_grants(
                livekit_api.VideoGrants(
                    room_join=True,
                    room=scoped_room,
                    can_publish=True,
                    can_subscribe=True,
                )
            )
            return token.to_jwt()
        except Exception as e:
            logger.error(f"Failed to create voice token: {e}")
            return None

    async def create_room(self, room_name: str, tenant_id: str) -> dict:
        if not self._available:
            return {"created": False, "error": "LiveKit not configured"}

        scoped_room = self._scoped(tenant_id, room_name)
        lk_api = self._make_api()
        if not lk_api:
            return {"created": False, "error": "LiveKit API init failed"}

        try:
            room = await lk_api.room.create_room(
                livekit_api.CreateRoomRequest(
                    name=scoped_room,
                    empty_timeout=300,
                    max_participants=2,
                )
            )
            return {"room_name": room.name, "sid": room.sid, "created": True}
        except Exception as e:
            logger.error(f"Failed to create room: {e}")
            return {"created": False, "error": str(e)}
        finally:
            await lk_api.aclose()

    async def dispatch_agent(self, room_name: str, tenant_id: str) -> dict:
        if not self._available:
            return {"created": False, "error": "LiveKit not configured"}

        scoped_room = self._scoped(tenant_id, room_name)
        lk_api = self._make_api()
        if not lk_api:
            return {"created": False, "error": "LiveKit API init failed"}

        try:
            existing = await lk_api.agent_dispatch.list_dispatch(room_name=scoped_room)
            for dispatch in existing:
                if getattr(dispatch, "agent_name", "") == settings.livekit_agent_name:
                    return {"created": True, "dispatch_id": dispatch.id, "existing": True}

            dispatch = await lk_api.agent_dispatch.create_dispatch(
                livekit_api.CreateAgentDispatchRequest(
                    agent_name=settings.livekit_agent_name,
                    room=scoped_room,
                    metadata=tenant_id,
                )
            )
            return {"created": True, "dispatch_id": dispatch.id, "existing": False}
        except Exception as e:
            logger.error(f"Failed to dispatch agent: {e}")
            return {"created": False, "error": str(e)}
        finally:
            await lk_api.aclose()

    async def list_rooms(self, tenant_id: str) -> list:
        if not self._available:
            return []

        lk_api = self._make_api()
        if not lk_api:
            return []

        try:
            response = await lk_api.room.list_rooms(livekit_api.ListRoomsRequest())
            prefix = f"{tenant_id}_"
            return [
                {"name": r.name, "participants": r.num_participants, "sid": r.sid}
                for r in response.rooms
                if r.name.startswith(prefix)
            ]
        except Exception as e:
            logger.error(f"Failed to list rooms: {e}")
            return []
        finally:
            await lk_api.aclose()

    async def delete_room(self, room_name: str, tenant_id: str) -> bool:
        if not self._available:
            return False

        scoped_room = self._scoped(tenant_id, room_name)
        lk_api = self._make_api()
        if not lk_api:
            return False

        try:
            await lk_api.room.delete_room(
                livekit_api.DeleteRoomRequest(room=scoped_room)
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete room: {e}")
            return False
        finally:
            await lk_api.aclose()


livekit_manager = LiveKitManager()
