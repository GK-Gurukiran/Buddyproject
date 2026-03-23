"""
Voice AI pipeline using LiveKit Agents framework.
Handles VAD (Voice Activity Detection), STT, LLM processing, and TTS.
"""

import logging
from typing import Optional

from app.agents.graph import agent_graph
from app.memory.mem0_client import memory_manager
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class VoicePipeline:
    """
    Manages the voice conversation pipeline:
    Audio In -> VAD -> STT -> LLM Agent Graph -> TTS -> Audio Out

    This class is designed to be used with LiveKit Agents framework.
    For production, you would run this as a separate worker process.
    """

    def __init__(self, tenant_id: str, user_id: str, session_id: str):
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.session_id = session_id

    async def process_transcript(self, transcript: str) -> str:
        """
        Process a speech-to-text transcript through the agent graph.
        Returns the text response to be converted to speech.
        """
        try:
            # Retrieve relevant memories for context
            memories = await memory_manager.search_memory(
                query=transcript,
                user_id=self.user_id,
                tenant_id=self.tenant_id,
                limit=3,
            )
            memory_context = "\n".join([m["content"] for m in memories]) if memories else ""

            # Run through agent graph
            initial_state = {
                "messages": [],
                "current_agent": "supervisor",
                "user_input": transcript,
                "research_results": None,
                "scrape_results": None,
                "final_response": None,
                "tenant_id": self.tenant_id,
                "user_id": self.user_id,
                "session_id": self.session_id,
                "memory_context": memory_context,
                "sources": [],
                "next_agent": None,
            }

            result = await agent_graph.ainvoke(initial_state)
            response = result.get("final_response", "I'm sorry, I couldn't process that.")

            # Store the interaction in memory
            await memory_manager.add_memory(
                content=f"User said (voice): {transcript}\nAssistant replied: {response[:500]}",
                user_id=self.user_id,
                tenant_id=self.tenant_id,
                session_id=self.session_id,
                metadata={"source": "voice"},
            )

            return response

        except Exception as e:
            logger.error(f"Voice pipeline error: {e}")
            return "I encountered an error processing your request. Please try again."


def create_livekit_worker_entrypoint():
    """
    Creates the entrypoint script content for running the LiveKit agent worker.
    This would be run as a separate process: python -m app.voice.worker

    Returns the code that should be placed in worker.py for production use.
    """
    return '''
"""
LiveKit Agent Worker - Run as a separate process.
Usage: python -m app.voice.worker

This worker connects to LiveKit and handles real-time voice interactions
using the Silero VAD, OpenAI STT/TTS, and the multi-agent graph.
"""

import asyncio
import logging
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import openai as lk_openai
from livekit.plugins import silero

from app.config import get_settings
from app.voice.pipeline import VoicePipeline

settings = get_settings()
logger = logging.getLogger("voice-worker")


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

    # Initialize voice pipeline
    pipeline = VoicePipeline(
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
    )

    # Create the voice pipeline agent
    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=lk_openai.STT(api_key=settings.openai_api_key),
        llm=lk_openai.LLM(api_key=settings.openai_api_key, model=settings.openai_model),
        tts=lk_openai.TTS(api_key=settings.openai_api_key),
    )

    agent.start(ctx.room, participant)
    await agent.say("Hello! How can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
            ws_url=settings.livekit_url,
        ),
    )
'''
