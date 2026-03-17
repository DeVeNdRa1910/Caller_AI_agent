"""
media_stream.py — Twilio WebSocket real-time audio pipeline (multi-tenant + RAG).

STREAMING ARCHITECTURE:

  OLD (sequential, ~6-17s per turn from your logs):
    STT done → LLM finishes entirely → TTS REST finishes → stream to Twilio

  NEW (parallel streaming, target ~1.2s to first audio):
    STT done → LLM starts streaming
                ↓ first sentence ready (~0.3s)
                Sarvam WS generates audio chunk
                ↓ first audio chunk (~0.8s)
                Twilio receives audio → user hears response
                (LLM still generating sentence 2 meanwhile)

  The function llm_to_tts_stream_chunks() in tts.py is an async generator
  that runs three coroutines in parallel:
    A) Read Groq stream tokens → asyncio.Queue
    B) Batch tokens into sentences → send to Sarvam WS
    C) Read Sarvam WS audio events → yield each MP3 chunk immediately

  media_stream.py iterates the generator, converts each chunk to
  8kHz mulaw, and sends to Twilio without waiting for the full response.
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
from tts import llm_to_tts_stream_chunks, synthesise_text, synthesise_text_bytes, AUDIO_DIR
from rag_pipeline import retrieve_context, build_rag_system_prompt

log = logging.getLogger(__name__)

VAD_FRAME_MS         = 20
VAD_SAMPLE_RATE      = 8000
VAD_FRAME_BYTES      = int(VAD_SAMPLE_RATE * 2 * VAD_FRAME_MS / 1000)
MIN_SPEECH_FRAMES    = 10
MIN_SILENCE_MS       = 500
MAX_RECORD_MS        = 10000
POST_TTS_COOLDOWN_MS = 600

_MAX_HISTORY_TURNS = 20

AGENT_BEHAVIOUR_PROMPT = """=== SCRIPT RULES — READ THIS FIRST ===

LANGUAGE — ABSOLUTE RULE:
You MUST write every single word in ROMAN SCRIPT (English letters).
NEVER output Devanagari, Gujarati, or any non-Latin characters.
Hinglish example: write "Theek hai, aapka ghar 2nd floor par hai."
NOT: "ठीक है, आपका घर 2nd floor पर है।"
Roman script only — Devanagari output breaks the voice system.

LENGTH — ABSOLUTE RULE:
Maximum 2 short sentences per reply. This is a live phone call.
Never write more than 2 sentences. No lists, bullets, or markdown.

=== YOUR ROLE ===
You are an AI voice agent on a LIVE OUTBOUND PHONE CALL.
All knowledge about this company, its services, and the call flow
comes ONLY from the Knowledge Base (KB) provided below.
You have no other knowledge. Do not invent information.
If something is not in the KB, say: "Main team se confirm karke bataungi."

=== CALL FLOW ===
The KB contains a section called "Mandatory Call Flow" with numbered steps.
Follow those steps IN ORDER. One step at a time.
- Check conversation history to know your current step.
- Ask ONLY the current step's question.
- Move to the next step ONLY after the customer answers the current one.
- NEVER re-ask an already answered question.
- NEVER repeat the opening greeting — it was already spoken.
- If unclear, ask the customer to repeat ONCE.
- If asked a question mid-flow, answer briefly from KB then return to current step.
"""


def _is_garbage_stt(text: str) -> bool:
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


def _mp3_chunk_to_mulaw_8k(mp3_bytes: bytes) -> bytes:
    from pydub import AudioSegment
    seg = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
    seg = seg.set_frame_rate(8000).set_channels(1)
    return audioop.lin2ulaw(seg.raw_data, 2)


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

        self._buffer          = bytearray()
        self._pcm_buffer      = bytearray()
        self._speech_frames   = 0
        self._silence_frames  = 0
        self._is_speech       = False
        self._record_start_ms: float | None = None

        self._cooldown_until_ms: float = 0.0
        self._extra_cooldown_ms: float = POST_TTS_COOLDOWN_MS

        self._ready_event  = asyncio.Event()
        self._ready_event.set()
        self._trigger_lock = asyncio.Lock()

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

        if self._is_speech and self._record_start_ms is not None:
            if (now_ms - self._record_start_ms) >= MAX_RECORD_MS:
                log.info("VAD hard cap (%dms) — forcing STT", MAX_RECORD_MS)
                await self._trigger_stt(ws)
                return

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
            self._ready_event.clear()

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

    # ── Core pipeline ──────────────────────────────────────────────────────────

    async def _process_audio(self, ws: WebSocket, mulaw_audio: bytes):
        t0 = time.perf_counter()
        try:
            # 1. Decode + validate
            pcm = audioop.ulaw2lin(mulaw_audio, 2)
            if len(pcm) < 4800:
                log.debug("Audio too short (%d bytes) — skipping", len(pcm))
                return

            wav_bytes = _pcm_to_wav_bytes(pcm, VAD_SAMPLE_RATE)

            # 2. STT
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

            # 3. RAG
            history  = self.call_history.get(self.call_sid or "", [])
            is_first = len(history) == 0
            context  = ""

            if self.tenant_id:
                rag_query = (
                    "mandatory call flow steps language greeting good time "
                    "confirm service household floor lift vehicle quotation survey address"
                    if is_first else user_text
                )
                context = await retrieve_context(
                    self.tenant_id, rag_query,
                    top_k=15 if is_first else 8,
                    max_chars=4000,
                )
                log.info("RAG: %d chars", len(context)) if context else \
                log.warning("RAG: NO context for tenant=%s", self.tenant_id)

            # 4. System prompt
            system_prompt = build_rag_system_prompt(AGENT_BEHAVIOUR_PROMPT, context)
            system_prompt += (
                "\n\n[CALL STATE: Opening greeting already spoken. "
                "Customer responding to language preference. "
                "You are on Step 1. Do NOT greet again.]"
                if is_first else
                "\n\n[CALL STATE: Call in progress. "
                "Check conversation history for current step. Continue from that step only.]"
            )

            # 5. LLM — stream=True
            messages = [{"role": "system", "content": system_prompt}]
            for msg in history[-(_MAX_HISTORY_TURNS * 2):]:
                messages.append(msg)
            messages.append({"role": "user", "content": user_text})

            t_llm = time.perf_counter()
            groq_stream = await self.groq_client.chat.completions.create(
                model       = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                messages    = messages,
                max_tokens  = 80,
                temperature = 0.0,
                stream      = True,
                tool_choice = "none",
            )
            log.info("⏱  LLM stream started: %.3fs", time.perf_counter() - t_llm)

            # 6. Stream LLM tokens → TTS → Twilio simultaneously
            #
            # llm_to_tts_stream_chunks() yields MP3 chunks as Sarvam produces them.
            # We convert and send each chunk to Twilio without waiting for the full response.
            # User hears first audio ~0.8s after LLM starts, not after it finishes.

            total_bytes = 0
            first_chunk = True
            ws_failed   = False

            async for mp3_chunk in llm_to_tts_stream_chunks(groq_stream):
                if first_chunk:
                    log.info("⏱  First audio chunk: %.3fs", time.perf_counter() - t0)
                    first_chunk = False

                total_bytes += len(mp3_chunk)

                try:
                    mulaw = await asyncio.to_thread(_mp3_chunk_to_mulaw_8k, mp3_chunk)
                except Exception as e:
                    log.warning("MP3→mulaw failed: %s", e)
                    continue

                for i in range(0, len(mulaw), 160):
                    b64 = base64.b64encode(mulaw[i: i + 160]).decode("ascii")
                    try:
                        await ws.send_text(json.dumps({
                            "event":     "media",
                            "streamSid": self.stream_sid,
                            "media":     {"payload": b64},
                        }))
                    except Exception as e:
                        log.warning("Twilio WS send failed: %s", e)
                        ws_failed = True
                        break
                    await asyncio.sleep(0.005)

                if ws_failed:
                    break

            # 7. Cooldown
            playback_est_ms = (total_bytes / 10_000) * 1000
            self._extra_cooldown_ms = max(POST_TTS_COOLDOWN_MS, playback_est_ms * 0.3)

            log.info("⏱  Total pipeline: %.3fs | audio: %d bytes",
                     time.perf_counter() - t0, total_bytes)

            # 8. Record reply in history (non-streaming call, runs after audio sent)
            ai_reply = await self._get_reply_for_history(messages)
            log.info("🤖 AGENT: %s", ai_reply)

            if self.call_sid and ai_reply:
                hist = self.call_history.setdefault(self.call_sid, [])
                hist.append({"role": "user",      "content": user_text})
                hist.append({"role": "assistant",  "content": ai_reply})
                if len(hist) > _MAX_HISTORY_TURNS * 2:
                    self.call_history[self.call_sid] = hist[-(_MAX_HISTORY_TURNS * 2):]

        except Exception as e:
            log.exception("_process_audio error: %s", e)
            try:
                await self._send_error_tts(ws)
            except Exception:
                pass
        finally:
            self._cooldown_until_ms = time.monotonic() * 1000 + self._extra_cooldown_ms
            self._ready_event.set()

    async def _get_reply_for_history(self, messages: list) -> str:
        """
        Small non-streaming LLM call to get the text reply for conversation history.
        Runs AFTER audio is already streamed to user so it doesn't add latency.
        """
        try:
            resp = await self.groq_client.chat.completions.create(
                model       = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                messages    = messages,
                max_tokens  = 80,
                temperature = 0.0,
                stream      = False,
                tool_choice = "none",
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            log.warning("History LLM call failed: %s", e)
            return "[streamed response]"

    async def _send_error_tts(self, ws: WebSocket):
        text = "Sorry, ek technical issue aa gaya. Ek moment please."
        try:
            mp3 = await synthesise_text_bytes(text)
            if not mp3:
                fname = await synthesise_text(text)
                with open(os.path.join(AUDIO_DIR, fname), "rb") as f:
                    mp3 = f.read()
            if mp3:
                mulaw = await asyncio.to_thread(_mp3_chunk_to_mulaw_8k, mp3)
                for i in range(0, len(mulaw), 160):
                    b64 = base64.b64encode(mulaw[i: i + 160]).decode("ascii")
                    await ws.send_text(json.dumps({
                        "event":     "media",
                        "streamSid": self.stream_sid,
                        "media":     {"payload": b64},
                    }))
                    await asyncio.sleep(0.01)
        except Exception as e:
            log.warning("Error TTS also failed: %s", e)


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