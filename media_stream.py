"""
media_stream.py — Twilio WebSocket real-time audio pipeline (multi-tenant + RAG).

DESIGN PHILOSOPHY — FULLY DOCUMENT-DRIVEN:
  - Zero hardcoded business logic (no company names, cities, questions, steps).
  - Every decision the agent makes comes from the tenant's knowledge document
    retrieved via RAG.
  - The system prompt instructs the LLM to read the "Mandatory Call Flow" section
    from the knowledge base and follow it step by step.
  - Adding a new tenant = upload their document. No code changes ever needed.

HOW THE FLOW WORKS:
  1. Call connects → opening greeting already played by main.py (/call-user).
  2. User speaks → STT → retrieve fresh RAG context → LLM generates reply.
  3. LLM is forced by system prompt to:
       a) Track which step it is on (using conversation history).
       b) Only ask the current step's question.
       c) Never jump ahead or repeat answered questions.
  4. TTS → stream audio back to Twilio.

FIXES RETAINED FROM PREVIOUS VERSION:
  - asyncio.Event for atomic processing flag.
  - History size cap (_MAX_HISTORY_TURNS).
  - tenant_id from Twilio customParameters (not query string).
  - VAD hard cap (MAX_RECORD_MS) to force STT on long utterances.
  - synthesise_to_bytes uses REST cache path (no WS lock contention).
  - System prompt rebuilt when tenant_id arrives from customParameters.
"""

import asyncio
import audioop
import base64
import io
import json
import logging
import os
import time
import wave

import webrtcvad
from fastapi import WebSocket
from groq import AsyncGroq

from dotenv import load_dotenv
load_dotenv()

from stt import transcribe_audio_async
from tts import synthesise_text_bytes, synthesise_text, AUDIO_DIR
from rag_pipeline import retrieve_context, build_rag_system_prompt

log = logging.getLogger(__name__)

# ── VAD / recording constants ──────────────────────────────────────────────────
VAD_FRAME_MS         = 20
VAD_SAMPLE_RATE      = 8000
VAD_FRAME_BYTES      = int(VAD_SAMPLE_RATE * 2 * VAD_FRAME_MS / 1000)
MIN_SPEECH_FRAMES    = 10      # ~200ms before treating as real speech
MIN_SILENCE_MS       = 500     # end of utterance after 500ms silence
MAX_RECORD_MS        = 10000   # force STT after 10s regardless of silence
POST_TTS_COOLDOWN_MS = 600     # suppress echo for 600ms after TTS finishes

_MAX_HISTORY_TURNS = 20        # cap conversation history to prevent memory growth

# ── Agent behaviour prompt ─────────────────────────────────────────────────────
# This is the ONLY behaviour instruction in code.
# ALL factual content (company name, steps, questions, services) comes from RAG.
# This prompt tells the LLM HOW to use the knowledge base, not WHAT to say.

AGENT_BEHAVIOUR_PROMPT = """You are an AI voice agent on a LIVE OUTBOUND PHONE CALL.

=== YOUR ONLY SOURCE OF KNOWLEDGE ===
Everything you say must come ONLY from the Knowledge Base (KB) provided.
You have NO other knowledge about this company, its services, prices, or process.
If something is not in the KB, say: "Main team se confirm karke aapko bataungi."

=== CALL FLOW ===
The KB contains a section called "Mandatory Call Flow" with numbered steps.
You MUST follow those steps IN ORDER, one at a time.

Rules:
- Check the conversation history to know which step you are currently on.
- Ask ONLY the current step's question. Do NOT jump ahead to the next step.
- Move to the next step ONLY after the customer has fully answered the current one.
- NEVER re-ask a question the customer has already answered.
- NEVER repeat the opening greeting — it was already spoken before the call connected.
- If the customer says something unclear, politely ask them to repeat ONCE.
- If the customer asks a question mid-flow, answer it briefly from the KB,
  then return to the current step's question.

=== LANGUAGE ===
- The first step in the call flow asks for language preference.
- Use whatever language the customer chooses for ALL subsequent replies.
- Respond in ROMANIZED script only — no Devanagari characters ever.
  Hindi/Hinglish must be written in English letters: write "Theek hai" not "ठीक है".

=== RESPONSE FORMAT ===
- Maximum 2 short sentences per reply. This is a phone call, not a text message.
- No bullet points, lists, or markdown — speak naturally.
- Do not echo back what the customer said. Just respond and continue.
"""


def _is_garbage_stt(text: str) -> bool:
    """Filter STT noise — repetitive or empty output."""
    if not text or not text.strip():
        return True
    words = text.strip().split()
    if len(words) >= 4 and len(set(words)) <= 2:
        return True
    return False


def _pcm_to_wav_bytes(pcm: bytes, sample_rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


def _mp3_to_mulaw_8k(mp3_bytes: bytes) -> bytes:
    from pydub import AudioSegment
    seg = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
    seg = seg.set_frame_rate(8000).set_channels(1)
    return audioop.lin2ulaw(seg.raw_data, 2)


# ── Main handler ───────────────────────────────────────────────────────────────

class MediaStreamHandler:
    def __init__(
        self,
        system_prompt: str,
        call_history: dict,
        groq_client: AsyncGroq,
        tenant_id: str | None = None,
        call_tenant_map: dict | None = None,
        get_tenant_prompt_fn=None,
    ):
        self.system_prompt        = system_prompt
        self.call_history         = call_history
        self.groq_client          = groq_client
        self.tenant_id            = tenant_id
        self.call_tenant_map      = call_tenant_map or {}
        self.get_tenant_prompt_fn = get_tenant_prompt_fn

        self.vad             = webrtcvad.Vad(3)
        self.stream_sid: str | None = None
        self.call_sid:   str | None = None

        # VAD state
        self._buffer          = bytearray()
        self._pcm_buffer      = bytearray()
        self._speech_frames   = 0
        self._silence_frames  = 0
        self._is_speech       = False
        self._record_start_ms: float | None = None

        self._cooldown_until_ms: float = 0.0
        self._extra_cooldown_ms: float = POST_TTS_COOLDOWN_MS

        # Atomic processing flag — prevents concurrent STT/LLM/TTS runs
        self._ready_event  = asyncio.Event()
        self._ready_event.set()
        self._trigger_lock = asyncio.Lock()

    # ── WebSocket lifecycle ────────────────────────────────────────────────────

    async def handle_connection(self, websocket: WebSocket):
        await websocket.accept()
        try:
            async for message in websocket.iter_text():
                await self._process_message(websocket, message)
        except Exception as e:
            log.exception("Media stream connection error: %s", e)

    async def _process_message(self, ws: WebSocket, message: str):
        try:
            data  = json.loads(message)
            event = data.get("event")
            if event == "start":
                await self._on_start(data)
            elif event == "media":
                await self._on_media(ws, data)
            elif event == "stop":
                log.info("Stream stopped — call_sid=%s tenant=%s",
                         self.call_sid, self.tenant_id)
        except Exception as e:
            log.warning("_process_message error: %s", e)

    async def _on_start(self, data: dict):
        self.stream_sid = data.get("streamSid")
        start           = data.get("start", {})
        self.call_sid   = start.get("callSid")

        # Twilio strips query params from WebSocket URLs in production.
        # tenant_id is passed as a <Parameter> tag inside <Stream> and arrives here.
        custom_params = start.get("customParameters", {})
        if not self.tenant_id:
            self.tenant_id = custom_params.get("tenant_id")
            if self.tenant_id:
                log.info("tenant_id from customParameters: %s", self.tenant_id)
                if self.get_tenant_prompt_fn:
                    try:
                        self.system_prompt = await self.get_tenant_prompt_fn(self.tenant_id)
                        log.info("System prompt rebuilt for tenant=%s", self.tenant_id)
                    except Exception as e:
                        log.warning("System prompt rebuild failed: %s", e)
            else:
                log.warning("tenant_id missing — RAG disabled. customParameters=%s",
                            custom_params)

        if self.call_sid and self.tenant_id:
            self.call_tenant_map[self.call_sid] = self.tenant_id

        self._reset_vad()
        log.info("Stream started — call_sid=%s tenant=%s", self.call_sid, self.tenant_id)

    # ── VAD / media handling ───────────────────────────────────────────────────

    async def _on_media(self, ws: WebSocket, data: dict):
        if not self._ready_event.is_set():
            self._reset_vad()
            return

        now_ms = time.monotonic() * 1000
        if now_ms < self._cooldown_until_ms:
            self._reset_vad()
            return

        media = data.get("media", {})
        if media.get("track") == "outbound":
            return

        payload = media.get("payload")
        if not payload:
            return

        mulaw_chunk = base64.b64decode(payload)
        self._buffer.extend(mulaw_chunk)
        pcm = audioop.ulaw2lin(mulaw_chunk, 2)
        self._pcm_buffer.extend(pcm)

        # Hard cap — force STT if utterance exceeds MAX_RECORD_MS
        if self._is_speech and self._record_start_ms is not None:
            if (now_ms - self._record_start_ms) >= MAX_RECORD_MS:
                log.info("VAD hard cap (%dms) — forcing STT", MAX_RECORD_MS)
                await self._trigger_stt(ws)
                return

        # Frame-level VAD
        while len(self._pcm_buffer) >= VAD_FRAME_BYTES:
            frame            = bytes(self._pcm_buffer[:VAD_FRAME_BYTES])
            self._pcm_buffer = self._pcm_buffer[VAD_FRAME_BYTES:]
            is_speech        = self.vad.is_speech(frame, VAD_SAMPLE_RATE)

            if is_speech:
                if not self._is_speech:
                    self._record_start_ms = now_ms
                self._speech_frames  += 1
                self._silence_frames  = 0
                self._is_speech       = True
            else:
                self._silence_frames += 1
                if self._is_speech and self._speech_frames >= MIN_SPEECH_FRAMES:
                    if self._silence_frames * VAD_FRAME_MS >= MIN_SILENCE_MS:
                        await self._trigger_stt(ws)
                        return

    async def _trigger_stt(self, ws: WebSocket):
        async with self._trigger_lock:
            if not self._ready_event.is_set():
                return
            self._ready_event.clear()   # atomic: mark as busy

        audio = bytes(self._buffer)
        self._reset_vad()
        asyncio.create_task(self._process_audio(ws, audio))

    def _reset_vad(self):
        self._buffer          = bytearray()
        self._pcm_buffer      = bytearray()
        self._speech_frames   = 0
        self._silence_frames  = 0
        self._is_speech       = False
        self._record_start_ms = None

    # ── STT → RAG → LLM → TTS pipeline ───────────────────────────────────────

    async def _process_audio(self, ws: WebSocket, mulaw_audio: bytes):
        t0 = time.perf_counter()
        try:
            # ── 1. Decode + validate ───────────────────────────────────────
            pcm = audioop.ulaw2lin(mulaw_audio, 2)
            if len(pcm) < 4800:
                log.debug("Audio too short (%d bytes) — skipping", len(pcm))
                return

            wav_bytes = _pcm_to_wav_bytes(pcm, VAD_SAMPLE_RATE)

            # ── 2. STT ────────────────────────────────────────────────────
            try:
                user_text = await asyncio.wait_for(
                    transcribe_audio_async(
                        audio_bytes  = wav_bytes,
                        content_type = "audio/wav",
                        filename     = "audio.wav",
                    ),
                    timeout=8.0,
                )
            except asyncio.TimeoutError:
                log.warning("STT timeout (8s) — dropping utterance")
                return

            log.info("STT %.3fs: %r", time.perf_counter() - t0, (user_text or "")[:80])

            if not user_text or not user_text.strip():
                return
            if _is_garbage_stt(user_text):
                log.info("STT garbage filtered: %r", user_text[:60])
                return

            user_text = user_text.strip()
            log.info("👤 USER: %s", user_text)

            # ── 3. RAG ────────────────────────────────────────────────────
            history  = self.call_history.get(self.call_sid or "", [])
            is_first = len(history) == 0

            context = ""
            if self.tenant_id:
                if is_first:
                    # First turn: load full call flow + company info
                    rag_query = "mandatory call flow steps company name service agent opening"
                    top_k     = 15
                else:
                    # Later turns: targeted retrieval on what user said
                    rag_query = user_text
                    top_k     = 8

                context = await retrieve_context(
                    self.tenant_id, rag_query, top_k=top_k, max_chars=4000
                )
                if context:
                    log.info("RAG: %d chars for query=%r", len(context), rag_query[:50])
                else:
                    log.warning("RAG: NO context for tenant=%s query=%r",
                                self.tenant_id, rag_query[:50])
            else:
                log.warning("No tenant_id — RAG disabled for this call")

            # ── 4. System prompt = behaviour rules + RAG knowledge base ───
            system_prompt = build_rag_system_prompt(AGENT_BEHAVIOUR_PROMPT, context)

            # Inject call-state hint so LLM knows where it is in the flow
            if is_first:
                system_prompt += (
                    "\n\n[CALL STATE: The opening greeting has already been spoken. "
                    "The customer is responding to the language preference question. "
                    "You are on Step 1 of the Mandatory Call Flow. "
                    "Do NOT greet again. Respond to what the customer said and "
                    "then proceed to Step 2.]"
                )
            else:
                system_prompt += (
                    "\n\n[CALL STATE: The call is in progress. "
                    "Check the conversation history to determine the current step. "
                    "Continue from that step — do NOT repeat or skip earlier steps.]"
                )

            # ── 5. Assemble messages ───────────────────────────────────────
            messages = [{"role": "system", "content": system_prompt}]
            for msg in history[-(_MAX_HISTORY_TURNS * 2):]:
                messages.append(msg)
            messages.append({"role": "user", "content": user_text})

            # ── 6. LLM ────────────────────────────────────────────────────
            t_llm    = time.perf_counter()
            response = await self.groq_client.chat.completions.create(
                model       = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                messages    = messages,
                max_tokens  = 150,
                temperature = 0.1,
                stream      = False,
                tool_choice = "none",
            )
            ai_reply = (response.choices[0].message.content or "").strip()
            log.info("LLM %.3fs: %r", time.perf_counter() - t_llm, ai_reply[:80])

            if not ai_reply:
                ai_reply = "Sorry, main samajh nahi payi. Kya aap dobara bol sakte hain?"

            # ── 7. TTS ────────────────────────────────────────────────────
            t_tts     = time.perf_counter()
            mp3_bytes = await self._synthesize_bytes(ai_reply)
            log.info("TTS %.3fs", time.perf_counter() - t_tts)

            # ── 8. Stream audio to Twilio ──────────────────────────────────
            total_bytes = 0
            if mp3_bytes:
                total_bytes = len(mp3_bytes)
                await self._stream_mp3_to_twilio(ws, mp3_bytes)

            # ── 9. Cooldown proportional to audio length ───────────────────
            playback_est_ms = (total_bytes / 10_000) * 1000
            self._extra_cooldown_ms = max(POST_TTS_COOLDOWN_MS, playback_est_ms * 0.3)

            log.info("🤖 AGENT: %s", ai_reply)

            # ── 10. Append to history AFTER successful pipeline ────────────
            if self.call_sid:
                hist = self.call_history.setdefault(self.call_sid, [])
                hist.append({"role": "user",      "content": user_text})
                hist.append({"role": "assistant",  "content": ai_reply})
                if len(hist) > _MAX_HISTORY_TURNS * 2:
                    self.call_history[self.call_sid] = hist[-(_MAX_HISTORY_TURNS * 2):]

            log.info("⏱  Total pipeline: %.3fs", time.perf_counter() - t0)

        except Exception as e:
            log.exception("_process_audio error: %s", e)
            try:
                await self._send_tts_safe(
                    ws, "Sorry, ek technical issue aa gaya. Ek moment please."
                )
            except Exception:
                pass
        finally:
            self._cooldown_until_ms = time.monotonic() * 1000 + self._extra_cooldown_ms
            self._ready_event.set()   # unblock VAD for next utterance

    # ── TTS helpers ────────────────────────────────────────────────────────────

    async def _synthesize_bytes(self, text: str) -> bytes | None:
        """Get MP3 bytes — cache-first, REST fallback."""
        try:
            mp3 = await synthesise_text_bytes(text)
            if mp3:
                return mp3
            fname    = await synthesise_text(text)
            filepath = os.path.join(AUDIO_DIR, fname)
            with open(filepath, "rb") as f:
                data = f.read()
            try:
                os.remove(filepath)
            except OSError:
                pass
            return data
        except Exception as e:
            log.warning("_synthesize_bytes error: %s", e)
            return None

    async def _send_tts_safe(self, ws: WebSocket, text: str) -> bool:
        mp3 = await self._synthesize_bytes(text)
        if not mp3:
            return False
        return await self._stream_mp3_to_twilio(ws, mp3)

    async def _stream_mp3_to_twilio(self, ws: WebSocket, mp3_bytes: bytes) -> bool:
        """Convert MP3 → 8k mulaw → send in 160-byte chunks over WebSocket."""
        try:
            mulaw = await asyncio.to_thread(_mp3_to_mulaw_8k, mp3_bytes)
        except Exception as e:
            log.warning("MP3→mulaw failed: %s", e)
            return False

        for i in range(0, len(mulaw), 160):
            b64 = base64.b64encode(mulaw[i: i + 160]).decode("ascii")
            try:
                await ws.send_text(json.dumps({
                    "event":     "media",
                    "streamSid": self.stream_sid,
                    "media":     {"payload": b64},
                }))
            except Exception as e:
                log.warning("WS send failed: %s", e)
                return False
            await asyncio.sleep(0.01)
        return True


# ── Entry point ────────────────────────────────────────────────────────────────

async def handle_media_stream(
    websocket:           WebSocket,
    system_prompt:       str,
    call_history:        dict,
    groq_client:         AsyncGroq,
    tenant_id:           str | None = None,
    call_tenant_map:     dict | None = None,
    get_tenant_prompt_fn             = None,
):
    handler = MediaStreamHandler(
        system_prompt        = system_prompt,
        call_history         = call_history,
        groq_client          = groq_client,
        tenant_id            = tenant_id,
        call_tenant_map      = call_tenant_map,
        get_tenant_prompt_fn = get_tenant_prompt_fn,
    )
    await handler.handle_connection(websocket)