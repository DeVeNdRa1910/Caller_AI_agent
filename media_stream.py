"""
media_stream.py — Plivo WebSocket handler for real-time bidirectional audio.

Plivo Media Streams protocol differences vs Twilio:
  ┌─────────────────────────┬───────────────────────┬─────────────────────────────┐
  │ Concept                 │ Twilio                │ Plivo                       │
  ├─────────────────────────┼───────────────────────┼─────────────────────────────┤
  │ Stream identifier       │ streamSid             │ streamId                    │
  │ Call identifier         │ start.callSid         │ start.callId                │
  │ Inbound audio event     │ event="media"         │ event="media"  (same ✅)    │
  │ Outbound audio event    │ event="media"         │ event="playAudio"           │
  │ Outbound payload fmt    │ {event,streamSid,     │ {event:"playAudio",         │
  │                         │  media:{payload:b64}} │  media:{contentType,        │
  │                         │                       │   sampleRate,payload:b64}}  │
  │ Stop event              │ event="stop"          │ event="stop"   (same ✅)    │
  │ Start event             │ event="start"         │ event="start"  (same ✅)    │
  │ Audio codec (inbound)   │ mulaw 8kHz            │ mulaw 8kHz     (same ✅)    │
  │ Audio codec (outbound)  │ mulaw 8kHz            │ mulaw 8kHz     (same ✅)    │
  └─────────────────────────┴───────────────────────┴─────────────────────────────┘

TTS codec note:
  tts.py already outputs mulaw 8kHz WAV (TTS_CODEC="mulaw", TTS_SAMPLE_RATE=8000).
  So we no longer need _mp3_to_mulaw_8k() — the bytes from synthesise_text_bytes()
  are already the correct format. We strip the WAV header (44 bytes) to get raw
  mulaw PCM before sending, which is what Plivo expects as the payload.

Flow:
  Plivo sends mulaw 8kHz → VAD detects speech end → STT → LLM stream
  → TTS per sentence (mulaw WAV) → strip WAV header → send chunks via playAudio event
"""

import asyncio
import audioop
import base64
import io
import json
import logging
import os
import re
import struct
import time
import wave

import webrtcvad
from fastapi import WebSocket
from groq import AsyncGroq

from dotenv import load_dotenv
load_dotenv()

from stt import transcribe_audio_async
from tts import synthesise_text, synthesise_text_bytes, AUDIO_DIR

log = logging.getLogger(__name__)

# ── VAD config (unchanged) ────────────────────────────────────────────────────
VAD_FRAME_MS    = 20
VAD_SAMPLE_RATE = 8000
VAD_FRAME_BYTES = int(VAD_SAMPLE_RATE * 2 * VAD_FRAME_MS / 1000)
MIN_SPEECH_FRAMES = 10
MIN_SILENCE_MS    = 350

SENTENCE_END    = re.compile(r"[।.?!]\s*")
MAX_BUFFER_CHARS = 120

# WAV header is always 44 bytes for standard PCM/mulaw WAV files.
# We strip it before sending raw mulaw payload to Plivo.
WAV_HEADER_BYTES = 44


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _pcm_to_wav_bytes(pcm: bytes, sample_rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


def _strip_wav_header(wav_bytes: bytes) -> bytes:
    """
    Strip the 44-byte WAV header and return raw mulaw PCM bytes.
    Validates the RIFF header before stripping; falls back to slice if invalid.
    """
    if len(wav_bytes) <= WAV_HEADER_BYTES:
        return wav_bytes
    if wav_bytes[:4] == b"RIFF":
        # Parse actual data chunk offset for robustness
        try:
            pos = 12
            while pos + 8 <= len(wav_bytes):
                chunk_id   = wav_bytes[pos:pos+4]
                chunk_size = struct.unpack_from("<I", wav_bytes, pos+4)[0]
                if chunk_id == b"data":
                    return wav_bytes[pos+8:]
                pos += 8 + chunk_size
        except Exception:
            pass
        return wav_bytes[WAV_HEADER_BYTES:]
    # Not a WAV file — return as-is (shouldn't happen with our TTS)
    return wav_bytes


# ─────────────────────────────────────────────────────────────────────────────
# HANDLER
# ─────────────────────────────────────────────────────────────────────────────

class MediaStreamHandler:

    def __init__(self, system_prompt: str, call_history: dict, groq_client: AsyncGroq):
        self.system_prompt = system_prompt
        self.call_history  = call_history
        self.groq_client   = groq_client
        self.vad           = webrtcvad.Vad(2)

        # Plivo identifiers
        self.stream_id : str | None = None   # was: stream_sid  (Twilio: streamSid)
        self.call_id   : str | None = None   # was: call_sid    (Twilio: start.callSid)

        # Audio / VAD state
        self._buffer        = bytearray()
        self._pcm_buffer    = bytearray()
        self._speech_frames = 0
        self._silence_frames= 0
        self._is_speech     = False
        self._processing    = False
        self._lock          = asyncio.Lock()

    # ── Public entry point ────────────────────────────────────────────────────

    async def handle_connection(self, websocket: WebSocket):
        await websocket.accept()
        try:
            async for message in websocket.iter_text():
                await self._process_message(websocket, message)
        except Exception as e:
            log.exception("Media stream error: %s", e)

    # ── Message router ────────────────────────────────────────────────────────

    async def _process_message(self, ws: WebSocket, message: str):
        try:
            data  = json.loads(message)
            event = data.get("event")

            if event == "start":
                # Plivo start payload:
                #   { "event": "start",
                #     "streamId": "...",          ← was streamSid in Twilio
                #     "start": { "callId": "..." } ← was callSid in Twilio
                #   }
                self.stream_id = data.get("streamId")              # Plivo key
                start          = data.get("start", {})
                self.call_id   = start.get("callId")               # Plivo key

                # Reset state for new call
                self._buffer         = bytearray()
                self._pcm_buffer     = bytearray()
                self._speech_frames  = 0
                self._silence_frames = 0
                self._is_speech      = False
                self._processing     = False
                log.info("Media stream start stream_id=%s call_id=%s",
                         self.stream_id, self.call_id)

            elif event == "media":
                # Inbound audio from Plivo — same structure as Twilio ✅
                # { "event": "media", "media": { "payload": "<b64 mulaw>" } }
                await self._on_media(ws, data)

            elif event == "stop":
                # Plivo fires this when the call ends / stream disconnects
                log.info("Media stream stop (call_id=%s)", self.call_id)

            else:
                log.debug("Unknown media stream event: %s", event)

        except Exception as e:
            log.warning("Process message error: %s", e)

    # ── Inbound audio / VAD ───────────────────────────────────────────────────

    async def _on_media(self, ws: WebSocket, data: dict):
        if self._processing:
            # Drop audio while we're already generating a response
            self._buffer         = bytearray()
            self._pcm_buffer     = bytearray()
            self._speech_frames  = 0
            self._silence_frames = 0
            return

        payload = data.get("media", {}).get("payload")
        if not payload:
            return

        mulaw_chunk = base64.b64decode(payload)
        self._buffer.extend(mulaw_chunk)

        # Convert mulaw → 16-bit PCM for VAD
        pcm = audioop.ulaw2lin(mulaw_chunk, 2)
        self._pcm_buffer.extend(pcm)

        while len(self._pcm_buffer) >= VAD_FRAME_BYTES:
            frame    = bytes(self._pcm_buffer[:VAD_FRAME_BYTES])
            self._pcm_buffer = self._pcm_buffer[VAD_FRAME_BYTES:]
            is_speech = self.vad.is_speech(frame, VAD_SAMPLE_RATE)

            if is_speech:
                self._speech_frames  += 1
                self._silence_frames  = 0
                self._is_speech       = True
            else:
                self._silence_frames += 1
                if (
                    self._is_speech
                    and self._speech_frames >= MIN_SPEECH_FRAMES
                    and self._silence_frames * VAD_FRAME_MS >= MIN_SILENCE_MS
                ):
                    async with self._lock:
                        if self._processing:
                            return
                        self._processing = True

                    audio_to_process      = bytes(self._buffer)
                    self._buffer          = bytearray()
                    self._pcm_buffer      = bytearray()
                    self._speech_frames   = 0
                    self._silence_frames  = 0
                    self._is_speech       = False
                    asyncio.create_task(self._process_audio(ws, audio_to_process))
                    return

    # ── STT → LLM → TTS pipeline ─────────────────────────────────────────────

    async def _process_audio(self, ws: WebSocket, mulaw_audio: bytes):
        t0 = time.perf_counter()
        try:
            pcm = audioop.ulaw2lin(mulaw_audio, 2)
            if len(pcm) < 8000:
                log.debug("Audio too short, skipping")
                return

            wav_bytes = _pcm_to_wav_bytes(pcm, VAD_SAMPLE_RATE)
            user_text = await transcribe_audio_async(
                audio_bytes  = wav_bytes,
                content_type = "audio/wav",
                filename     = "audio.wav",
            )
            log.info("STT %.3fs: %s", time.perf_counter() - t0, (user_text or "")[:80])

            if not (user_text and user_text.strip()):
                await self._send_tts(ws, "कृपया दोबारा बोलें।")
                return

            history  = self.call_history.get(self.call_id or "", [])[-4:]
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(history)
            messages.append({"role": "user", "content": user_text.strip()})

            if self.call_id:
                self.call_history.setdefault(self.call_id, []).append(
                    {"role": "user", "content": user_text.strip()}
                )

            full_reply = ""
            async for sentence in self._stream_llm_sentences(messages):
                if not sentence.strip():
                    continue
                full_reply += sentence.strip() + " "
                if not await self._send_tts_safe(ws, sentence.strip()):
                    break   # WS closed — call ended

            full_reply = full_reply.strip()
            if self.call_id and full_reply:
                self.call_history.setdefault(self.call_id, []).append(
                    {"role": "assistant", "content": full_reply}
                )
            log.info("Media pipeline %.3fs", time.perf_counter() - t0)

        except Exception as e:
            log.exception("Process audio error: %s", e)
            try:
                await self._send_tts(ws, "माफ़ कीजिये, अभी तकनीकी समस्या है।")
            except Exception:
                pass
        finally:
            self._processing = False

    # ── LLM streaming ─────────────────────────────────────────────────────────

    async def _stream_llm_sentences(self, messages: list):
        """
        Stream LLM tokens; yield complete sentences (on ।.?!).
        First audio plays ~2–3s after user stops speaking.
        """
        stream = await self.groq_client.chat.completions.create(
            model       = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            messages    = messages,
            max_tokens  = 128,
            temperature = 0.1,
            stream      = True,
            tool_choice = "none",
        )
        buffer = ""
        async for chunk in stream:
            delta   = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            buffer += delta
            m = SENTENCE_END.search(buffer)
            if m:
                sentence = buffer[: m.end()].strip()
                buffer   = buffer[m.end():]
                if sentence:
                    yield sentence
            elif len(buffer.strip()) >= MAX_BUFFER_CHARS:
                sentence = buffer.strip()
                buffer   = ""
                if sentence:
                    yield sentence
        if buffer.strip():
            yield buffer.strip()

    # ── Outbound audio to Plivo ───────────────────────────────────────────────

    async def _send_mulaw_to_ws(self, ws: WebSocket, mulaw: bytes) -> bool:
        """
        Send raw mulaw bytes to Plivo via the 'playAudio' event.

        Plivo outbound audio event (DIFFERENT from Twilio):
          Twilio: { "event": "media",     "streamSid": "...", "media": {"payload": b64} }
          Plivo:  { "event": "playAudio", "media": {"contentType": "audio/x-mulaw",
                                                     "sampleRate": 8000,
                                                     "payload": b64} }

        Plivo does NOT use streamId in the outbound media event.
        contentType must be "audio/x-mulaw" and sampleRate must match the
        stream negotiation (8000 Hz, set in the <Stream> XML element).
        """
        chunk_size = 160  # 20ms of mulaw at 8kHz
        for i in range(0, len(mulaw), chunk_size):
            chunk = mulaw[i: i + chunk_size]
            b64   = base64.b64encode(chunk).decode("ascii")
            try:
                await ws.send_text(json.dumps({
                    "event": "playAudio",           
                    "media": {
                        "contentType": "audio/x-mulaw",   
                        "sampleRate":  8000,               
                        "payload":     b64,
                    },
                }))
            except RuntimeError as e:
                if "websocket.send" in str(e) or "already completed" in str(e):
                    log.debug("WS closed during send, stopping")
                    return False
                raise
            except Exception as e:
                log.debug("WS send error: %s", e)
                return False
            await asyncio.sleep(0.02)
        return True

    async def _send_tts_safe(self, ws: WebSocket, text: str) -> bool:
        """
            Synthesise text → mulaw WAV bytes → strip WAV header → send to Plivo.
            Returns False if WS is closed (call ended).

            tts.py already outputs mulaw 8kHz WAV, so no MP3→mulaw conversion needed.
            We just strip the 44-byte WAV header to get raw mulaw PCM.
        """
        wav_bytes = await synthesise_text_bytes(text)



        if wav_bytes:
            mulaw = _strip_wav_header(wav_bytes)
            return await self._send_mulaw_to_ws(ws, mulaw)

        # Fallback: read from file if in-memory path failed
        filename = await synthesise_text(text)
        filepath = os.path.join(AUDIO_DIR, filename)
        try:
            with open(filepath, "rb") as f:
                wav_bytes = f.read()
            mulaw = _strip_wav_header(wav_bytes)
            return await self._send_mulaw_to_ws(ws, mulaw)
        finally:
            try:
                os.remove(filepath)
            except OSError:
                pass

    async def _send_tts(self, ws: WebSocket, text: str):
        """Fire-and-forget TTS send. Swallows errors if WS already closed."""
        await self._send_tts_safe(ws, text)


async def handle_media_stream(
    websocket    : WebSocket,
    system_prompt: str,
    call_history : dict,
    groq_client  : AsyncGroq,
):
    handler = MediaStreamHandler(system_prompt, call_history, groq_client)
    await handler.handle_connection(websocket)
