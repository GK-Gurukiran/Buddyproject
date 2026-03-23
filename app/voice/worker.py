"""
LiveKit Agent Worker - Run as a separate process.
Usage: python -m app.voice.worker

This worker connects to LiveKit and handles real-time voice interactions
using the Silero VAD, OpenAI STT/TTS, and the multi-agent graph.

Requires: Docker LiveKit server running + livekit-agents installed.
"""

import asyncio
import logging
import sys

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger("voice-worker")

# Check if LiveKit agents SDK is available
try:
    from livekit.agents import (
        AutoSubscribe,
        JobContext,
        WorkerOptions,
        cli,
    )
    from livekit.agents.pipeline import VoicePipelineAgent
    from livekit.plugins import openai as lk_openai
    from livekit.plugins import silero
    LIVEKIT_AGENTS_AVAILABLE = True
except ImportError:
    LIVEKIT_AGENTS_AVAILABLE = False
    logger.warning("LiveKit agents SDK not installed. Voice worker cannot start.")


if LIVEKIT_AGENTS_AVAILABLE:
    async def entrypoint(ctx: JobContext):
        """Main entrypoint for each voice session."""

        # Extract tenant and user info from room metadata
        room_name = ctx.room.name
        parts = room_name.split("_", 1)
        tenant_id = parts[0] if len(parts) > 1 else "default"
        session_id = parts[1] if len(parts) > 1 else room_name

        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

        participant = await ctx.wait_for_participant()
        user_id = participant.identity

        logger.info(f"Voice session started: tenant={tenant_id}, user={user_id}")

        # Determine which API key to use for STT/TTS
        api_key = settings.openrouter_api_key or settings.openai_api_key
        base_url = settings.openrouter_base_url if settings.openrouter_api_key else None
        model = settings.openrouter_model if settings.openrouter_api_key else settings.openai_model

        # Create the voice pipeline agent
        stt_kwargs = {"api_key": api_key}
        tts_kwargs = {"api_key": api_key}
        llm_kwargs = {"api_key": api_key, "model": model}

        if base_url:
            llm_kwargs["base_url"] = base_url

        agent = VoicePipelineAgent(
            vad=silero.VAD.load(),
            stt=lk_openai.STT(**stt_kwargs),
            llm=lk_openai.LLM(**llm_kwargs),
            tts=lk_openai.TTS(**tts_kwargs),
        )

        agent.start(ctx.room, participant)
        await agent.say("Hello! How can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    if not LIVEKIT_AGENTS_AVAILABLE:
        print("ERROR: LiveKit agents SDK not installed.")
        print("Install with: pip install livekit-agents")
        sys.exit(1)

    if not settings.livekit_api_key:
        print("ERROR: LIVEKIT_API_KEY not configured.")
        print("Set LIVEKIT_API_KEY and LIVEKIT_API_SECRET in .env")
        sys.exit(1)

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
            ws_url=settings.livekit_url,
        ),
    )
