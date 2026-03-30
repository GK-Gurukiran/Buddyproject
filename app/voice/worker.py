"""
LiveKit Agent Worker — Deepgram STT + Cartesia TTS + Silero VAD + GPT-4o + Mem0 memory.
"""
import asyncio
import json
import logging
import sys

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import openai, deepgram, cartesia, silero

from app.config import get_settings
from app.memory.mem0_client import memory_manager

# Configure logging to ensure output is captured in Docker logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
settings = get_settings()
logger = logging.getLogger("voice-worker")


def mask(key: str) -> str:
    return f"{key[:4]}...{key[-4:]}" if (key and len(key) > 8) else "****"


def build_llm_config() -> tuple[dict, str]:
    if settings.llm_provider == "openrouter" and settings.openrouter_api_key:
        return (
            {
                "model": settings.openrouter_model,
                "api_key": settings.openrouter_api_key,
                "base_url": settings.openrouter_base_url,
            },
            "openrouter",
        )

    return (
        {
            "model": settings.openai_model,
            "api_key": settings.openai_api_key,
        },
        "openai",
    )


def _message_text(msg: llm.ChatMessage) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(part for part in content if isinstance(part, str))
    return str(content)


def _tts_language(language: str) -> str:
    if not language:
        return "en"
    return language.split("-", 1)[0].lower() or "en"


async def entrypoint(ctx: JobContext):
    llm_config, llm_backend = build_llm_config()

    logger.info(f"--- New Job Assigned: {ctx.job.id} ---")
    logger.info(f"Connecting to room: {ctx.room.name}")
    logger.info(f"LIVEKIT_URL={settings.livekit_url}")
    logger.info(f"LLM_PROVIDER={settings.llm_provider}")
    logger.info(f"LLM_BACKEND={llm_backend}")
    logger.info(f"OPENAI_API_KEY={mask(settings.openai_api_key)}")
    logger.info(f"OPENROUTER_API_KEY={mask(settings.openrouter_api_key)}")
    logger.info(f"DEEPGRAM_API_KEY={mask(settings.deepgram_api_key)}")
    logger.info(f"CARTESIA_API_KEY={mask(settings.cartesia_api_key)}")

    try:
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info(f"Connected to room: {ctx.room.name}")
        
        participant = await ctx.wait_for_participant()
        logger.info(f"Participant joined: {participant.identity}")

        # Parse room name: "{tenant_id}_voice_{session_id}"
        parts = ctx.room.name.split("_")
        tenant_id = parts[0] if len(parts) >= 3 else "default"
        session_id = parts[-1] if len(parts) >= 3 else ctx.room.name
        user_id = participant.identity

        # Parse language from participant metadata
        language = "en-US"
        if participant.metadata:
            try:
                meta = json.loads(participant.metadata)
                language = meta.get("language", "en-US")
            except Exception:
                pass

        logger.info(f"Session: tenant={tenant_id} user={user_id} session={session_id} lang={language}")

        def publish_transcript(role: str, text: str) -> None:
            if not text:
                return
            try:
                ctx.room.local_participant.publish_data(
                    json.dumps({"role": role, "text": text}),
                    reliable=True,
                    topic="transcript",
                )
            except Exception as e:
                logger.warning("Transcript publish failed (non-fatal): %s", e)

        # --- Memory injection callback ---
        async def before_llm_cb(agent: VoicePipelineAgent, chat_ctx: llm.ChatContext):
            """Inject relevant Mem0 memories before each LLM call."""
            last_user_msg = next(
                (m for m in reversed(chat_ctx.messages) if m.role == "user"), None
            )
            if not last_user_msg:
                return

            try:
                last_user_text = _message_text(last_user_msg)
                memories = await memory_manager.search_memory(
                    query=last_user_text,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    limit=3,
                )
                if memories:
                    memory_text = "\n".join(f"- {m['content']}" for m in memories)
                    # Inject as a system message just before the last user message
                    chat_ctx.messages.insert(
                        -1,
                        llm.ChatMessage.create(
                            role="system",
                            text=f"Relevant context from memory:\n{memory_text}",
                        ),
                    )
                    logger.info(f"Injected {len(memories)} memories into context")
            except Exception as e:
                logger.warning(f"Memory lookup failed (non-fatal): {e}")

        # --- Build the agent ---
        initial_ctx = llm.ChatContext().append(
            role="system",
            text=(
                "You are Buddy, a helpful AI voice assistant. "
                "Keep responses concise and natural for voice — no bullet points or markdown. "
                "You remember past conversations with this user."
            ),
        )

        logger.info("Initializing VoicePipelineAgent...")
        agent = VoicePipelineAgent(
            vad=silero.VAD.load(),
            stt=deepgram.STT(
                api_key=settings.deepgram_api_key,
                language=language,
                interim_results=True,
            ),
            llm=openai.LLM(
                **llm_config,
            ),
            tts=cartesia.TTS(
                api_key=settings.cartesia_api_key,
                model="sonic-2",
                language=_tts_language(language),
                encoding="pcm_s16le",
                sample_rate=24000,
                voice="794f9389-aac1-45b6-b726-9d9369183238",
            ),
            chat_ctx=initial_ctx,
            before_llm_cb=before_llm_cb,
        )

        last_user_text = ""

        @agent.on("user_speech_committed")
        def _on_user_speech(msg: llm.ChatMessage) -> None:
            nonlocal last_user_text
            last_user_text = _message_text(msg)
            logger.info("User speech committed: %s", last_user_text)
            publish_transcript("user", last_user_text)

        @agent.on("agent_speech_committed")
        def _on_agent_speech(msg: llm.ChatMessage) -> None:
            agent_text = _message_text(msg)
            logger.info("Agent speech committed: %s", agent_text)
            publish_transcript("assistant", agent_text)

            async def _save_memory() -> None:
                if not last_user_text:
                    return
                try:
                    await memory_manager.add_memory(
                        content=f"User (voice): {last_user_text}\nAssistant: {agent_text[:500]}",
                        user_id=user_id,
                        tenant_id=tenant_id,
                        session_id=session_id,
                        metadata={"source": "voice"},
                    )
                except Exception as e:
                    logger.warning(f"Memory save failed (non-fatal): {e}")

            asyncio.create_task(_save_memory())

        logger.info("Starting agent...")
        agent.start(ctx.room, participant)
        
        await agent.say(
            "Hi! I'm Buddy, your voice assistant. How can I help you today?",
            allow_interruptions=False,
        )
        logger.info("Greeting spoken. Agent ready.")

    except Exception as e:
        logger.error(f"Error in voice entrypoint: {e}", exc_info=True)
        if ctx.room:
            await ctx.room.disconnect()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=settings.livekit_agent_name,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
            ws_url=settings.livekit_url,
        )
    )
