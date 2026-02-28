"""
=============================================================================
Inbound Voice Agent — LiveKit Agents + Twilio SIP + Supabase Logging
=============================================================================
What this file does:
  1. Listens for inbound phone calls routed via Twilio SIP → LiveKit Cloud
  2. Runs a fully-streaming STT → LLM → TTS voice pipeline
  3. Supports barge-in (caller can interrupt the agent mid-sentence)
  4. Enforces a maximum call duration to prevent runaway API costs
  5. Logs every call (transcript, caller number, duration) to Supabase

Pipeline flow:
  Caller speaks → Deepgram STT → OpenAI LLM (streaming) → ElevenLabs TTS → Caller hears response

=============================================================================
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

# LiveKit Agents framework
from livekit import agents, rtc
from livekit.agents import AgentSession, JobContext, WorkerOptions, cli
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.voice import VoicePipelineAgent

# STT / LLM / TTS / VAD plugins
from livekit.plugins import deepgram, elevenlabs, openai, silero

# Supabase Python client
from supabase import Client, create_client

# Twilio request validator (for webhook security)
from twilio.request_validator import RequestValidator

# ---------------------------------------------------------------------------
# Load environment variables from .env file
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup — readable timestamps, easy to tail on a VPS
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("voice-agent")

# ---------------------------------------------------------------------------
# Read all configuration from environment variables
# NOTHING is hardcoded — all secrets come from .env
# ---------------------------------------------------------------------------

# LiveKit (required for the agent to connect to your LiveKit Cloud room)
LIVEKIT_URL = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]

# AI Service keys
DEEPGRAM_API_KEY = os.environ["DEEPGRAM_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]

# ElevenLabs voice settings
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Default: Rachel

# Twilio (for webhook validation — keeps bad actors from spoofing calls)
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]

# Supabase
SUPABASE_URL = os.environ["SUPABASE_URL"]          # e.g. https://supabase.yourdomain.com
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]  # Service role key (not anon key)

# Cost protection — maximum call duration in seconds (default 10 minutes)
MAX_CALL_DURATION_SECONDS = int(os.getenv("MAX_CALL_DURATION_SECONDS", "600"))

# The system prompt that defines the agent's personality and purpose
AGENT_SYSTEM_PROMPT = os.getenv(
    "AGENT_SYSTEM_PROMPT",
    "You are a helpful, friendly voice assistant answering inbound phone calls. "
    "Be concise — phone callers prefer short, clear answers. "
    "If you don't know something, say so honestly rather than guessing.",
)

# OpenAI model to use
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Supabase client — initialized once at startup
# ---------------------------------------------------------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# ---------------------------------------------------------------------------
# Supabase logging helper
# ---------------------------------------------------------------------------
async def log_call_to_supabase(
    caller_number: str,
    duration_seconds: float,
    transcript: str,
) -> None:
    """
    Insert a call record into the Supabase `call_logs` table.

    This runs after the call ends. If it fails, we log the error but
    do NOT crash — the call has already completed successfully.

    Table schema expected in Supabase:
        call_logs (
            id              uuid default gen_random_uuid() primary key,
            caller_number   text,
            duration_seconds numeric,
            transcript      text,
            created_at      timestamptz default now()
        )
    """
    try:
        record = {
            "caller_number": caller_number,
            "duration_seconds": round(duration_seconds, 2),
            "transcript": transcript,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # supabase-py .insert() returns the inserted row(s)
        response = (
            supabase.table("call_logs")
            .insert(record)
            .execute()
        )

        logger.info(
            f"✅ Call logged to Supabase — caller: {caller_number}, "
            f"duration: {duration_seconds:.1f}s, "
            f"transcript length: {len(transcript)} chars"
        )

    except Exception as e:
        # IMPORTANT: We catch ALL exceptions here so a Supabase failure
        # never crashes the agent process or affects call quality.
        logger.error(f"❌ Failed to log call to Supabase: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# Transcript accumulator
# ---------------------------------------------------------------------------
class TranscriptCollector:
    """
    Collects all STT segments during a call and assembles them into
    a full readable transcript at the end.

    Each turn is labeled [CALLER] or [AGENT] for readability.
    """

    def __init__(self):
        self._entries: list[str] = []

    def add_caller(self, text: str) -> None:
        if text.strip():
            self._entries.append(f"[CALLER]: {text.strip()}")

    def add_agent(self, text: str) -> None:
        if text.strip():
            self._entries.append(f"[AGENT]: {text.strip()}")

    def get_full_transcript(self) -> str:
        return "\n".join(self._entries)


# ---------------------------------------------------------------------------
# Main agent entrypoint — called once per inbound call
# ---------------------------------------------------------------------------
async def entrypoint(ctx: JobContext) -> None:
    """
    This function is called by LiveKit Agents every time a new call room
    is created (i.e., every inbound phone call).

    ctx.room contains info about the LiveKit room for this call.
    The caller's SIP metadata (phone number) is in the room's participant info.
    """

    call_start_time = time.time()
    transcript = TranscriptCollector()

    # -----------------------------------------------------------------------
    # Extract caller phone number from SIP participant metadata
    # LiveKit passes Twilio SIP info via participant attributes/identity
    # -----------------------------------------------------------------------
    caller_number = "unknown"

    async def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        nonlocal caller_number
        # SIP calls from Twilio include the caller's number in participant identity
        # or in the SIP metadata attributes
        identity = participant.identity or ""
        attrs = participant.attributes or {}

        # Try SIP-specific attributes first (set by LiveKit SIP bridge)
        sip_from = attrs.get("sip.from", "") or attrs.get("sip_from", "")
        if sip_from:
            caller_number = sip_from
        elif identity.startswith("+") or identity.lstrip("+").isdigit():
            # Identity looks like a phone number
            caller_number = identity
        else:
            caller_number = identity or "unknown"

        logger.info(f"📞 Caller connected: {caller_number}")

    ctx.room.on("participant_connected", on_participant_connected)

    # -----------------------------------------------------------------------
    # Connect the agent to the LiveKit room
    # -----------------------------------------------------------------------
    await ctx.connect()
    logger.info(f"🔗 Agent connected to room: {ctx.room.name}")

    # -----------------------------------------------------------------------
    # Configure Silero VAD (Voice Activity Detection)
    #
    # VAD decides WHEN the caller is speaking vs. silent.
    # Getting these values right prevents the agent from:
    #   - Cutting the caller off mid-sentence (too aggressive)
    #   - Waiting too long after they finish (too lenient)
    # -----------------------------------------------------------------------
    vad = silero.VAD.load(
        # min_speech_duration: Minimum milliseconds of speech before VAD
        # considers it a real utterance (filters out clicks/noise).
        # Lower = more sensitive. Raise if you get false triggers.
        min_speech_duration=0.05,  # 50ms

        # min_silence_duration: How many milliseconds of silence after speech
        # before VAD says "the person finished speaking".
        # Lower = agent responds faster but may cut caller off.
        # Raise (e.g. 0.8–1.2) if callers complain about being interrupted.
        min_silence_duration=0.5,  # 500ms — good balance for phone calls

        # padding_duration: Extra audio padding added around speech segments.
        # Helps capture the very start and end of words.
        padding_duration=0.1,  # 100ms

        # activation_threshold: Confidence score (0–1) required to consider
        # audio as speech. Higher = less sensitive (fewer false positives).
        # Lower = more sensitive (catches quiet speech but more noise triggers).
        activation_threshold=0.5,

        # sample_rate: Must match the audio pipeline sample rate.
        # 8000 Hz is standard for telephone audio (Twilio/SIP).
        sample_rate=8000,
    )

    # -----------------------------------------------------------------------
    # Build the LLM with the system prompt
    # -----------------------------------------------------------------------
    llm = openai.LLM(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
    )

    initial_ctx = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=AGENT_SYSTEM_PROMPT,
            )
        ]
    )

    # -----------------------------------------------------------------------
    # Build the fully-streaming voice pipeline agent
    #
    # Key streaming features:
    #   - STT streams partial transcripts as caller speaks (Deepgram)
    #   - LLM streams tokens as they're generated (OpenAI)
    #   - TTS converts each sentence fragment to audio immediately (ElevenLabs)
    #   - Result: caller hears first words of response in ~500ms, not 3–5s
    #
    # Barge-in: allow_interruptions=True means if the caller speaks while
    #   the agent is talking, the agent STOPS IMMEDIATELY and listens.
    #   This is critical for natural conversation flow.
    # -----------------------------------------------------------------------
    agent = VoicePipelineAgent(
        # Speech-to-Text: Deepgram Nova-2 — best accuracy for phone calls
        stt=deepgram.STT(
            api_key=DEEPGRAM_API_KEY,
            model="nova-2-phonecall",  # Optimized model for telephone audio
            language="en",
            smart_format=True,         # Auto-punctuation, numbers, etc.
        ),

        # Language Model: OpenAI with streaming enabled by default
        llm=llm,

        # Text-to-Speech: ElevenLabs with streaming (sentence-by-sentence)
        tts=elevenlabs.TTS(
            api_key=ELEVENLABS_API_KEY,
            voice_id=ELEVENLABS_VOICE_ID,
            model="eleven_turbo_v2_5",  # Lowest latency ElevenLabs model
            # Streaming: ElevenLabs will send audio chunks as text arrives,
            # not wait for the full response — critical for low latency.
            streaming=True,
        ),

        # Voice Activity Detection: Silero (configured above)
        vad=vad,

        # Chat context (system prompt + conversation history)
        chat_ctx=initial_ctx,

        # BARGE-IN: If caller speaks while agent is talking, agent stops.
        # Set to False if you want the agent to always finish its sentence.
        allow_interruptions=True,

        # How quickly the agent responds after detecting end-of-speech.
        # Lower = faster response but risk of cutting off caller.
        # min_endpointing_delay matches the VAD silence duration.
        min_endpointing_delay=0.5,

        # Thinking indicator: play a filler sound ("hmm", "let me check...")
        # while waiting for LLM. Keeps silence from feeling like a dropped call.
        # Set to None to disable.
        # preemptive_synthesis=True,  # Uncomment to enable
    )

    # -----------------------------------------------------------------------
    # Intercept agent and caller speech to build transcript
    # -----------------------------------------------------------------------
    @agent.on("user_speech_committed")
    def on_user_speech(user_msg: agents.llm.ChatMessage) -> None:
        """Called when the STT finalizes a caller utterance."""
        text = user_msg.content if isinstance(user_msg.content, str) else str(user_msg.content)
        transcript.add_caller(text)
        logger.info(f"👤 Caller: {text}")

    @agent.on("agent_speech_committed")
    def on_agent_speech(agent_msg: agents.llm.ChatMessage) -> None:
        """Called when the agent finishes speaking a response."""
        text = agent_msg.content if isinstance(agent_msg.content, str) else str(agent_msg.content)
        transcript.add_agent(text)
        logger.info(f"🤖 Agent: {text}")

    # -----------------------------------------------------------------------
    # Maximum call duration enforcement (cost protection)
    #
    # If a call exceeds MAX_CALL_DURATION_SECONDS, the agent says goodbye
    # and disconnects. This prevents infinite loops or stuck calls from
    # consuming unbounded API credits.
    # -----------------------------------------------------------------------
    async def enforce_max_duration() -> None:
        await asyncio.sleep(MAX_CALL_DURATION_SECONDS)
        logger.warning(
            f"⏱️ Call exceeded maximum duration ({MAX_CALL_DURATION_SECONDS}s). "
            "Disconnecting to prevent cost overrun."
        )
        try:
            await agent.say(
                "I'm sorry, we've reached the maximum call duration. "
                "Please call back if you need further assistance. Goodbye!",
                allow_interruptions=False,
            )
        except Exception:
            pass
        await ctx.room.disconnect()

    duration_task = asyncio.create_task(enforce_max_duration())

    # -----------------------------------------------------------------------
    # Start the agent — it will now listen and respond until call ends
    # -----------------------------------------------------------------------
    agent.start(ctx.room)

    # Greet the caller immediately when they connect
    await agent.say(
        "Hello! Thank you for calling. How can I help you today?",
        allow_interruptions=True,
    )

    # -----------------------------------------------------------------------
    # Wait for the call to end (room disconnects when Twilio hangs up)
    # -----------------------------------------------------------------------
    try:
        await ctx.room.wait_for_disconnect()
    finally:
        # Cancel the max-duration watchdog if the call ended normally
        duration_task.cancel()

        # Calculate actual call duration
        call_duration = time.time() - call_start_time
        full_transcript = transcript.get_full_transcript()

        logger.info(
            f"📵 Call ended — caller: {caller_number}, "
            f"duration: {call_duration:.1f}s, "
            f"transcript lines: {len(full_transcript.splitlines())}"
        )

        # Log to Supabase — wrapped in try/except inside the function
        # so a Supabase failure never propagates up and crashes the process
        await log_call_to_supabase(
            caller_number=caller_number,
            duration_seconds=call_duration,
            transcript=full_transcript,
        )


# ---------------------------------------------------------------------------
# Run the worker — long-running process that waits for inbound calls
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            # agent_name ties this worker to a specific LiveKit dispatch rule
            # Must match what you configure in LiveKit Cloud console
            agent_name="inbound-voice-agent",
        )
    )
