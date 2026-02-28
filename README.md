# 📞 Inbound Voice Agent

A self-hosted AI voice agent that answers inbound phone calls using LiveKit Agents, Twilio SIP, and a fully-streaming STT → LLM → TTS pipeline. Every call is logged to your Supabase instance.

---

## Architecture Overview

```
Caller dials Twilio number
        ↓
Twilio SIP → LiveKit Cloud (SIP Bridge)
        ↓
LiveKit Room → This agent (agent.py on your VPS)
        ↓
Deepgram STT → OpenAI LLM → ElevenLabs TTS → Caller hears response
        ↓
Call ends → Supabase logs (transcript, duration, caller number)
```

---

## Supabase Table Setup

Before running the agent, create the `call_logs` table in your Supabase instance.

**Open Supabase → SQL Editor → New Query → paste and run:**

```sql
create table if not exists call_logs (
    id               uuid default gen_random_uuid() primary key,
    caller_number    text,
    duration_seconds numeric(10, 2),
    transcript       text,
    created_at       timestamptz default now()
);

-- Optional: Index for fast queries by date or caller
create index on call_logs (created_at desc);
create index on call_logs (caller_number);

-- Optional: Enable Row Level Security (recommended for production)
alter table call_logs enable row level security;

-- Allow service role full access (the agent uses service role key)
create policy "Service role full access" on call_logs
    for all
    using (true)
    with check (true);
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in every value:

```bash
cp .env.example .env
```

| Variable | Description | Where to Get It |
|----------|-------------|-----------------|
| `LIVEKIT_URL` | Your LiveKit Cloud WebSocket URL | [cloud.livekit.io](https://cloud.livekit.io) → Project → Settings → Keys |
| `LIVEKIT_API_KEY` | LiveKit API key | Same as above |
| `LIVEKIT_API_SECRET` | LiveKit API secret | Same as above |
| `DEEPGRAM_API_KEY` | Deepgram API key for STT | [console.deepgram.com](https://console.deepgram.com) → API Keys |
| `OPENAI_API_KEY` | OpenAI API key for LLM | [platform.openai.com](https://platform.openai.com) → API Keys |
| `OPENAI_MODEL` | OpenAI model name | `gpt-4o-mini` (cheap) or `gpt-4o` (best) |
| `ELEVENLABS_API_KEY` | ElevenLabs API key for TTS | [elevenlabs.io](https://elevenlabs.io) → Profile → API Key |
| `ELEVENLABS_VOICE_ID` | Voice ID for TTS | ElevenLabs Voice Library |
| `TWILIO_AUTH_TOKEN` | Twilio auth token | [console.twilio.com](https://console.twilio.com) → Account Info |
| `SUPABASE_URL` | Your Supabase instance URL | Self-hosted: `https://supabase.yourdomain.com`; Cloud: Settings → API |
| `SUPABASE_SERVICE_KEY` | Supabase service role key | Supabase → Settings → API → `service_role` key (**not** `anon`) |
| `MAX_CALL_DURATION_SECONDS` | Max call length before auto-hangup | Set to your expected max (default: `600` = 10 min) |
| `AGENT_SYSTEM_PROMPT` | Personality/purpose of the agent | Write your own or use the default |

---

## Installation

### 1. Clone / copy files to your VPS

```bash
mkdir ~/voice-agent && cd ~/voice-agent
# Copy agent.py, requirements.txt, .env.example, .gitignore here
cp .env.example .env
nano .env  # Fill in your values
```

### 2. Create a Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Silero VAD model (one-time)

```bash
python -c "from livekit.plugins import silero; silero.VAD.load()"
```

---

## LiveKit Cloud Setup

1. Go to [cloud.livekit.io](https://cloud.livekit.io) and create a project
2. Under **Settings → Keys**, create an API key/secret pair → put in `.env`
3. Under **SIP**, enable the SIP trunk feature
4. Create an **Inbound SIP Trunk** and connect it to your Twilio number (see Twilio section)
5. Create a **Dispatch Rule** that routes inbound SIP calls to agent name `inbound-voice-agent`

---

## Twilio SIP Setup

1. Log in to [console.twilio.com](https://console.twilio.com)
2. Buy or use an existing phone number
3. Go to **Elastic SIP Trunking** → Create a trunk
4. Under **Origination**, add your LiveKit SIP URI:
   ```
   sip:your-project.sip.livekit.cloud
   ```
5. Point your Twilio phone number at this SIP trunk
6. Copy your **Auth Token** from Account Info → put in `.env` as `TWILIO_AUTH_TOKEN`

---

## Running the Agent

### Development (foreground)

```bash
source venv/bin/activate
python agent.py dev
```

### Production (persistent background process)

Use `systemd` to keep the agent running after reboots and auto-restart on crashes:

```bash
sudo nano /etc/systemd/system/voice-agent.service
```

Paste:

```ini
[Unit]
Description=LiveKit Inbound Voice Agent
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/voice-agent
ExecStart=/home/ubuntu/voice-agent/venv/bin/python agent.py start
Restart=on-failure
RestartSec=5
EnvironmentFile=/home/ubuntu/voice-agent/.env

[Install]
WantedBy=multi-user.target
```

Then enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable voice-agent
sudo systemctl start voice-agent

# Check logs
sudo journalctl -u voice-agent -f
```

---

## VAD Tuning Guide

If callers are being cut off mid-sentence, open `agent.py` and increase `min_silence_duration`:

```python
min_silence_duration=0.8,  # was 0.5 — gives more pause before agent responds
```

If there's too much delay after the caller finishes speaking:

```python
min_silence_duration=0.3,  # faster response, but risk of interrupting
```

---

## Pre-Launch Checklist

- [ ] LiveKit Cloud account created, API keys in `.env`
- [ ] LiveKit SIP trunk enabled and Dispatch Rule pointing to `inbound-voice-agent`
- [ ] Twilio phone number purchased
- [ ] Twilio SIP trunk created, origination URI set to LiveKit SIP URI
- [ ] Deepgram account, API key in `.env`
- [ ] OpenAI account, API key in `.env`
- [ ] ElevenLabs account, API key + Voice ID in `.env`
- [ ] Supabase `call_logs` table created (run SQL above)
- [ ] Supabase URL + service role key in `.env`
- [ ] `MAX_CALL_DURATION_SECONDS` set to a sensible limit
- [ ] `AGENT_SYSTEM_PROMPT` customized for your use case
- [ ] Agent tested in dev mode (`python agent.py dev`)
- [ ] systemd service configured and running
- [ ] Test call placed — check `journalctl` logs and Supabase table for the record

---

## Cost Estimate (per minute of call)

| Service | Approx. Cost |
|---------|-------------|
| Deepgram Nova-2 | ~$0.0043/min |
| OpenAI GPT-4o-mini | ~$0.002–0.01/call (varies by tokens) |
| ElevenLabs Turbo v2.5 | ~$0.015/min |
| LiveKit Cloud | Check your plan |
| Twilio SIP | ~$0.004/min + number rental |

Set `MAX_CALL_DURATION_SECONDS` conservatively to protect against runaway calls.
