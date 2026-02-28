"""
=============================================================================
Inbound Voice Agent — LiveKit Agents v1.x + Twilio SIP + Supabase Logging
=============================================================================
Tridasa Rise | Presales & Sales Voice Agent | Nallagandla, Hyderabad

Compatible with: livekit-agents >= 1.0 (released April 2025)
Install:         pip install -r requirements.txt

What changed from v0.x:
  - VoicePipelineAgent is REMOVED → replaced by AgentSession + Agent subclass
  - ChatContext/ChatMessage imports moved → use Agent(instructions=...) instead
  - agent.start() → session.start()
  - agent.say() → session.generate_reply(instructions=...)
  - AgentServer + @server.rtc_session() is the new recommended pattern

Pipeline flow:
  Caller speaks
    → Deepgram STT (streaming)
    → OpenAI LLM (streaming tokens)
    → ElevenLabs TTS (streaming audio chunks)
    → Caller hears response
  Call ends → Supabase call_logs insert

=============================================================================
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# LiveKit Agents v1.x — correct imports based on official migration guide
# ---------------------------------------------------------------------------
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.agents import room_io                          # RoomOptions lives here in v1.x
from livekit.plugins import deepgram, elevenlabs, openai, silero

# Supabase Python client
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# Load environment variables from .env file
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Logging — timestamped, easy to tail with: journalctl -u voice-agent -f
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("voice-agent")

# ---------------------------------------------------------------------------
# Configuration — ALL values come from .env, nothing hardcoded
# ---------------------------------------------------------------------------

# LiveKit Cloud
LIVEKIT_URL        = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY    = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]

# AI services
DEEPGRAM_API_KEY   = os.environ["DEEPGRAM_API_KEY"]
OPENAI_API_KEY     = os.environ["OPENAI_API_KEY"]
ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]

# ElevenLabs voice — default is "Rachel", a clear and friendly voice
# Find other voice IDs at: https://elevenlabs.io/voice-library
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

# Supabase — use service role key (not anon key) for write access
SUPABASE_URL         = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

# Cost protection — agent hangs up after this many seconds (default: 10 min)
MAX_CALL_DURATION_SECONDS = int(os.getenv("MAX_CALL_DURATION_SECONDS", "600"))

# OpenAI model — gpt-4o-mini is cheapest, gpt-4o is most capable
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# System prompt — Tridasa Rise presales & sales agent
# Full conversation flow: Hook → Overview → Amenities → Location → Specs
#                         → Objection Handling → Close
# ---------------------------------------------------------------------------
AGENT_SYSTEM_PROMPT = os.getenv(
    "AGENT_SYSTEM_PROMPT",
    """
You are a professional presales and sales agent for Tridasa Rise, a premium
residential community in Nallagandla, Hyderabad. You answer inbound phone calls
from prospective buyers. Your goal is to understand their requirement, present
the project compellingly, handle objections confidently, and convert the call
into a site visit or availability check.

PERSONALITY & TONE:
- Warm, confident, and consultative — not pushy
- Speak in clear, concise sentences suited for a phone call
- Never read out bullet lists robotically — weave facts naturally into conversation
- Always listen carefully and respond to what the caller actually says

CONVERSATION FLOW — follow this sequence naturally:

1. OPENING (first 20 seconds):
   Greet the caller warmly. Introduce Tridasa Rise as a thoughtfully designed
   residential community in Nallagandla where modern comfort meets peaceful
   living — a premium low-density project offering space, lifestyle, and
   connectivity in one place.

2. PROJECT OVERVIEW — mention key facts naturally when relevant:
   - Land: 10.38 Acres | 7 Towers | 17 Floors | Only 92 units per acre (low density)
   - Clubhouse: 55,000 sq ft
   - Unit sizes: 1733 to 2751 sq ft
   - Configurations: 3 BHK and 4 BHK
   - Facing options: East, West, and North
   Use this line: "This is ideal for buyers who want spacious homes in a peaceful
   yet well-connected location."

3. LIFESTYLE & AMENITIES — present by category based on what the caller cares about:
   - Wellness: Yoga space, meditation zone, relaxation lounge, senior citizen outdoor
     gym, forest walk, scented gardens. Say: "You can literally unwind without stepping
     outside the community."
   - Sports & Fitness: Tennis, pickleball, basketball, cricket net, skating zone,
     outdoor fitness, walking track. Say: "It's like having a private sports club
     right at home."
   - Family & Social: Children's play area, kids activity zone, kids gardening, party
     lawns, open air theatre, banquet hall, guest rooms. Say: "It's designed for every
     age group — kids, adults, and seniors."
   - Work & Convenience: Co-working space, business centre, market zone, provision for
     supermarket, pharmacy, ATM, and creche. Say: "You can live, work, and manage daily
     needs without leaving the community."
   - Clubhouse: Gym, swimming pool, indoor games, mini theatre, badminton, squash,
     terrace party deck, hobby zone. Say: "This clubhouse alone feels like a premium
     resort membership."

4. LOCATION ADVANTAGES — share relevant points based on what the caller mentions:
   Schools (5-15 mins): Narayana Jr. College, Sadhana Infinity International School,
   Glendale, Meru, Bharatiya Vidya Bhavan's, Sancta Maria, Bachpan, Hyderabad Central
   University.
   Hospitals (5-30 mins): Citizens Hospitals 5 min, Previse 15 min, Continental 25 min,
   CARE and Star Hospitals 30 min.
   Transport: Lingampally Station 5 min, BHEL 10 min, Financial District 25 min,
   Airport 40 min.
   Shopping: Aparna Neo Mall 5 min, GSM and Sarath City 20 min, Inorbit 30 min.
   Close with: "You're perfectly connected to everything important, yet far enough from
   city chaos to enjoy peaceful living."

5. QUALITY & SPECIFICATIONS — mention only once interest is shown:
   RCC shear wall structure (seismic and wind resistant), premium vitrified flooring,
   anti-skid bathroom tiles, Kohler or equivalent sanitary fittings, uPVC windows with
   mesh, false ceiling in bathrooms, AC points in all bedrooms and living areas,
   3-phase power with prepaid meters, internet provision in every flat, EV charging per
   flat, DG backup, 24/7 security, fire sprinklers, centralized LPG.
   Say: "Every detail is planned not just for comfort, but for long-term convenience
   and safety."

6. OBJECTION HANDLING:
   Price concern: "This is a low-density premium project with large homes and a 55,000
   sq ft clubhouse — which is why it offers higher long-term value."
   Location doubt: "Nallagandla is one of the fastest-growing residential corridors
   because it combines peaceful surroundings with excellent connectivity."
   Competitor comparison: "Most projects compromise on either space, amenities, or
   connectivity. Tridasa Rise is one of the rare communities that offers all three."

7. CLOSING:
   Soft close: "Would you like me to share available floor plans based on your preferred
   size and facing?"
   Direct close: "Units are limited because of the low-density design. Shall I check
   current availability for you right now?"
   Follow-up: "When would you prefer a site visit — weekday or weekend?"

IMPORTANT RULES:
- Never make up prices, unit availability, or possession dates.
  Say: "I'll have our sales team confirm that for you."
- If you don't know something, say: "Let me connect you with our team who can give
  you the exact details."
- Keep each response under 4 sentences unless the caller asks for more detail.
- Always end with a question to keep the conversation moving forward.
""",
)

# ---------------------------------------------------------------------------
# Supabase client — created once at startup, reused for all calls
# ---------------------------------------------------------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# ---------------------------------------------------------------------------
# Supabase logging — ALWAYS wrapped in try/except
# A Supabase failure must never crash the agent process or affect call quality
# ---------------------------------------------------------------------------
async def log_call_to_supabase(
    caller_number: str,
    duration_seconds: float,
    transcript: str,
) -> None:
    """
    Insert one row into the Supabase call_logs table after a call ends.

    Required table schema (run in Supabase SQL Editor):
        create table call_logs (
            id               uuid default gen_random_uuid() primary key,
            caller_number    text,
            duration_seconds numeric(10, 2),
            transcript       text,
            created_at       timestamptz default now()
        );
    """
    try:
        record = {
            "caller_number": caller_number,
            "duration_seconds": round(duration_seconds, 2),
            "transcript": transcript,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        supabase.table("call_logs").insert(record).execute()
        logger.info(
            f"✅ Supabase logged — caller: {caller_number}, "
            f"duration: {duration_seconds:.1f}s, "
            f"transcript: {len(transcript)} chars"
        )
    except Exception as e:
        # Log everything we tried to save so it can be recovered manually
        logger.error(f"❌ Supabase insert failed: {e}", exc_info=True)
        logger.error(
            f"   UNSAVED — caller: {caller_number}, "
            f"duration: {duration_seconds:.1f}s, "
            f"transcript_len: {len(transcript)}"
        )


# ---------------------------------------------------------------------------
# Transcript accumulator — builds a readable conversation log per call
# ---------------------------------------------------------------------------
class TranscriptCollector:
    """
    Collects every utterance during a call (both caller and agent)
    and formats them into a clean labelled transcript at the end.
    """

    def __init__(self) -> None:
        self._lines: list[str] = []

    def add(self, role: str, text: str) -> None:
        """role should be 'CALLER' or 'AGENT'"""
        if text.strip():
            self._lines.append(f"[{role}]: {text.strip()}")

    def full(self) -> str:
        return "\n".join(self._lines)


# ---------------------------------------------------------------------------
# The Agent class — defines personality using v1.x API
#
# In v1.x, Agent holds the instructions (system prompt) and optional hooks.
# AgentSession (in entrypoint) holds the pipeline components: STT, LLM, TTS, VAD.
# ---------------------------------------------------------------------------
class TridasaRiseAgent(Agent):
    """
    Tridasa Rise inbound sales agent.

    The session (STT/LLM/TTS/VAD pipeline) is created in entrypoint().
    This class only needs to define instructions and lifecycle hooks.
    """

    def __init__(self) -> None:
        super().__init__(instructions=AGENT_SYSTEM_PROMPT)

    async def on_enter(self) -> None:
        """
        Called automatically when the agent becomes active in a session.
        We trigger the opening greeting here so the agent speaks first.

        Note: on_enter is async and must use await when calling session methods.
        """
        await self.session.generate_reply(
            instructions=(
                "Greet the caller warmly. Introduce yourself as a Tridasa Rise "
                "sales consultant. Keep it to 2-3 sentences maximum — this is "
                "the opening of a phone call."
            )
        )


# ---------------------------------------------------------------------------
# Entrypoint — called by LiveKit once per inbound call
# ---------------------------------------------------------------------------
async def entrypoint(ctx: JobContext) -> None:
    """
    LiveKit creates a new room for every inbound Twilio SIP call and calls
    this function. ctx.room gives access to the room and participants.
    """

    call_start_time = time.time()
    transcript = TranscriptCollector()
    caller_number = "unknown"

    # -----------------------------------------------------------------------
    # Connect the agent worker to the LiveKit room
    # -----------------------------------------------------------------------
    await ctx.connect()
    logger.info(f"🔗 Connected to room: {ctx.room.name}")

    # -----------------------------------------------------------------------
    # Extract caller phone number from SIP participant attributes
    # Twilio passes the caller's number via SIP headers, which LiveKit
    # exposes as participant attributes (sip.from) or identity.
    # -----------------------------------------------------------------------
    for participant in ctx.room.remote_participants.values():
        attrs = participant.attributes or {}
        # LiveKit SIP bridge sets sip.from to the caller's phone number
        sip_from = attrs.get("sip.from") or attrs.get("sip_from") or ""
        if sip_from:
            caller_number = sip_from
        elif participant.identity:
            caller_number = participant.identity
        break  # Only the first remote participant is the caller

    logger.info(f"📞 Caller: {caller_number}")

    # -----------------------------------------------------------------------
    # Silero VAD configuration
    #
    # VAD (Voice Activity Detection) decides when the caller is speaking
    # vs. silent. Getting this right prevents:
    #   - Cutting the caller off mid-sentence (min_silence_duration too low)
    #   - Awkward pauses before agent responds (min_silence_duration too high)
    #
    # See README → VAD Tuning Guide to adjust for your call quality.
    # -----------------------------------------------------------------------
    vad = silero.VAD.load(
        # Minimum audio length (seconds) to count as a real utterance.
        # Shorter bursts (clicks, background noise) are ignored.
        # Raise to 0.1 if you get false speech triggers from noise.
        min_speech_duration=0.05,

        # Silence gap (seconds) after speech before treating turn as done.
        # Lower = faster agent response. Raise to 0.8-1.2 if callers feel cut off.
        # 0.5s is a good starting point for telephone conversations.
        min_silence_duration=0.5,

        # Extra audio padding added at the START of detected speech (seconds).
        # Prevents clipping the very beginning of words.
        # Renamed from padding_duration in Silero v1.5.0+
        prefix_padding_duration=0.1,

        # Confidence score (0.0 to 1.0) to classify audio as speech.
        # Raise to 0.7 in noisy environments. Lower to 0.3 for quiet callers.
        activation_threshold=0.5,

        # 8000 Hz is the standard sample rate for telephone audio (Twilio/SIP).
        # Do not change this unless you know your audio source uses a different rate.
        sample_rate=8000,
    )

    # -----------------------------------------------------------------------
    # Build the AgentSession — the v1.x pipeline orchestrator
    #
    # AgentSession replaces VoicePipelineAgent from v0.x.
    # It manages the full STT → LLM → TTS chain, all streaming by default:
    #   - Deepgram streams partial transcripts as caller speaks
    #   - OpenAI streams LLM tokens as they generate
    #   - ElevenLabs streams audio sentence-by-sentence
    # Result: caller hears the first word of the response in ~400-600ms,
    # not after the full response is generated (which would take 3-5 seconds).
    #
    # allow_interruptions=True enables barge-in:
    #   If the caller speaks while the agent is talking, the agent STOPS
    #   immediately and listens. Critical for natural phone conversations.
    # -----------------------------------------------------------------------
    session = AgentSession(
        # Speech-to-Text: Deepgram Nova-2 optimized for telephone audio
        stt=deepgram.STT(
            api_key=DEEPGRAM_API_KEY,
            model="nova-2-phonecall",  # Best model for 8kHz phone calls
            language="en",
            smart_format=True,         # Auto-punctuation, number formatting
        ),

        # Language Model: OpenAI — streaming is on by default in v1.x
        llm=openai.LLM(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
        ),

        # Text-to-Speech: OpenAI TTS — reliable fallback, works on all paid plans
        # Switch back to ElevenLabs once quota/plan is confirmed working
        # Available voices: alloy, echo, fable, onyx, nova, shimmer
        # "nova" and "shimmer" are best for warm, conversational tone
        tts=openai.TTS(
            api_key=OPENAI_API_KEY,
            voice="nova",        # warm, clear female voice — good for sales
            model="tts-1",       # tts-1 = lowest latency, tts-1-hd = higher quality
        ),

        # Voice Activity Detection: Silero (configured above)
        vad=vad,

        # Barge-in: True = caller can interrupt agent mid-sentence.
        # Set to False if agent must always finish speaking before listening.
        allow_interruptions=True,

        # Seconds to wait after end-of-speech before sending to LLM.
        # Should match or be slightly above min_silence_duration in VAD.
        min_endpointing_delay=0.5,
    )

    # -----------------------------------------------------------------------
    # Capture speech events to build transcript
    # These fire every time a complete utterance is finalized by STT or TTS.
    # -----------------------------------------------------------------------
    @session.on("user_speech_committed")
    def on_user_speech(msg) -> None:
        """Fires when STT finalizes a caller utterance."""
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        transcript.add("CALLER", text)
        logger.info(f"👤 Caller: {text}")

    @session.on("agent_speech_committed")
    def on_agent_speech(msg) -> None:
        """Fires when the agent finishes speaking a complete response."""
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        transcript.add("AGENT", text)
        logger.info(f"🤖 Agent: {text}")

    # -----------------------------------------------------------------------
    # Max call duration watchdog — cost protection
    #
    # If a call runs longer than MAX_CALL_DURATION_SECONDS (default: 10 min),
    # the agent says goodbye and disconnects. This prevents runaway API costs
    # from stuck sessions, loops, or callers who leave the line open.
    # Configure the limit in .env: MAX_CALL_DURATION_SECONDS=600
    # -----------------------------------------------------------------------
    async def enforce_max_duration() -> None:
        await asyncio.sleep(MAX_CALL_DURATION_SECONDS)
        logger.warning(
            f"⏱️ Max call duration reached ({MAX_CALL_DURATION_SECONDS}s) — "
            "disconnecting to prevent cost overrun."
        )
        try:
            await session.generate_reply(
                instructions=(
                    "Apologize briefly and tell the caller that the maximum call "
                    "duration has been reached. Ask them to call back if they need "
                    "further help. Say goodbye warmly."
                )
            )
            # Small pause so the goodbye message can finish playing
            await asyncio.sleep(5)
        except Exception:
            pass
        await session.aclose()

    duration_task = asyncio.create_task(enforce_max_duration())

    # -----------------------------------------------------------------------
    # Start the session — connects the pipeline to the room and activates
    # the TridasaRiseAgent, which will fire on_enter() to greet the caller
    # -----------------------------------------------------------------------
    await session.start(
        room=ctx.room,
        agent=TridasaRiseAgent(),
        room_options=room_io.RoomOptions(
            # Audio-only — no video processing needed for phone calls
            audio_input=room_io.AudioInputOptions(),
        ),
    )

    # -----------------------------------------------------------------------
    # Wait until the session ends (caller hangs up → session closes)
    # In LiveKit Agents v1.x, ctx.room has no wait_for_disconnect().
    # Instead, AgentSession emits a "close" event — we wait on that.
    # -----------------------------------------------------------------------
    close_event = asyncio.Event()

    @session.on("close")
    def on_session_close(*args) -> None:
        close_event.set()

    try:
        await close_event.wait()
    finally:
        # Stop the cost watchdog if call ended before the limit
        duration_task.cancel()

        call_duration = time.time() - call_start_time
        full_transcript = transcript.full()

        logger.info(
            f"📵 Call ended — caller: {caller_number}, "
            f"duration: {call_duration:.1f}s, "
            f"transcript lines: {len(full_transcript.splitlines())}"
        )

        # Log to Supabase — errors are caught inside, will never crash here
        await log_call_to_supabase(
            caller_number=caller_number,
            duration_seconds=call_duration,
            transcript=full_transcript,
        )


# ---------------------------------------------------------------------------
# Run the agent worker — persistent long-running process
#
# Usage:
#   Development:  python agent.py dev
#   Production:   python agent.py start
#
# The worker connects to LiveKit Cloud and waits for inbound call rooms.
# agent_name must match the Dispatch Rule you create in LiveKit Cloud console.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="inbound-voice-agent",
        )
    )