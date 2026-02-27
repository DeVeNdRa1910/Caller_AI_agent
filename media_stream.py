"""
Media Streams handler — Twilio WebSocket for real-time bidirectional audio.

Reference: call-customer-support-system (GitHub). Enables ~2s "agent starts speaking"
by pushing first sentence as soon as it's ready instead of waiting for full reply.

Flow: Twilio sends mulaw 8kHz → we buffer, VAD detects speech end → STT → LLM stream
→ TTS per sentence → convert to mulaw → send chunks to Twilio.
"""

import asyncio
import audioop
import base64
import io
import json
import logging
import os
import re
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

VAD_FRAME_MS = 20
VAD_SAMPLE_RATE = 8000
VAD_FRAME_BYTES = int(VAD_SAMPLE_RATE * 2 * VAD_FRAME_MS / 1000)
MIN_SPEECH_FRAMES = 10
MIN_SILENCE_MS = 350
SENTENCE_END = re.compile(r"[।.?!]\s*")
# Safety: if LLM streams this many chars without sentence end, yield so first audio ~2–3s
MAX_BUFFER_CHARS = 120


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
    pcm = seg.raw_data
    return audioop.lin2ulaw(pcm, 2)


class MediaStreamHandler:
    def __init__(self, system_prompt: str, call_history: dict, groq_client: AsyncGroq):
        self.system_prompt = system_prompt
        self.call_history = call_history
        self.groq_client = groq_client
        self.vad = webrtcvad.Vad(2)
        self.stream_sid: str | None = None
        self.call_sid: str | None = None
        self._buffer = bytearray()
        self._pcm_buffer = bytearray()
        self._speech_frames = 0
        self._silence_frames = 0
        self._is_speech = False
        self._processing = False
        self._lock = asyncio.Lock()

    async def handle_connection(self, websocket: WebSocket):
        await websocket.accept()
        try:
            async for message in websocket.iter_text():
                await self._process_message(websocket, message)
        except Exception as e:
            log.exception("Media stream error: %s", e)

    async def _process_message(self, ws: WebSocket, message: str):
        try:
            data = json.loads(message)
            event = data.get("event")
            if event == "start":
                self.stream_sid = data.get("streamSid")
                start = data.get("start", {})
                self.call_sid = start.get("callSid")
                self._buffer = bytearray()
                self._pcm_buffer = bytearray()
                self._speech_frames = 0
                self._silence_frames = 0
                self._is_speech = False
                log.info("Media stream start call_sid=%s", self.call_sid)
            elif event == "media":
                await self._on_media(ws, data)
            elif event == "stop":
                log.info("Media stream stop")
        except Exception as e:
            log.warning("Process message error: %s", e)

    async def _on_media(self, ws: WebSocket, data: dict):
        if self._processing:
            self._buffer = bytearray()
            self._pcm_buffer = bytearray()
            self._speech_frames = 0
            self._silence_frames = 0
            return
        payload = data.get("media", {}).get("payload")
        if not payload:
            return
        mulaw_chunk = base64.b64decode(payload)
        self._buffer.extend(mulaw_chunk)
        pcm = audioop.ulaw2lin(mulaw_chunk, 2)
        self._pcm_buffer.extend(pcm)
        while len(self._pcm_buffer) >= VAD_FRAME_BYTES:
            frame = bytes(self._pcm_buffer[:VAD_FRAME_BYTES])
            self._pcm_buffer = self._pcm_buffer[VAD_FRAME_BYTES:]
            is_speech = self.vad.is_speech(frame, VAD_SAMPLE_RATE)
            if is_speech:
                self._speech_frames += 1
                self._silence_frames = 0
                self._is_speech = True
            else:
                self._silence_frames += 1
                if self._is_speech and self._speech_frames >= MIN_SPEECH_FRAMES:
                    if self._silence_frames * VAD_FRAME_MS >= MIN_SILENCE_MS:
                        async with self._lock:
                            if self._processing:
                                return
                            self._processing = True
                        audio_to_process = bytes(self._buffer)
                        self._buffer = bytearray()
                        self._pcm_buffer = bytearray()
                        self._speech_frames = 0
                        self._silence_frames = 0
                        self._is_speech = False
                        asyncio.create_task(self._process_audio(ws, audio_to_process))
                        return
            if len(self._pcm_buffer) < VAD_FRAME_BYTES:
                break

    async def _process_audio(self, ws: WebSocket, mulaw_audio: bytes):
        t0 = time.perf_counter()
        try:
            pcm = audioop.ulaw2lin(mulaw_audio, 2)
            if len(pcm) < 8000:
                log.debug("Audio too short, skip")
                return
            wav_bytes = _pcm_to_wav_bytes(pcm, VAD_SAMPLE_RATE)
            user_text = await transcribe_audio_async(
                audio_bytes=wav_bytes,
                content_type="audio/wav",
                filename="audio.wav",
            )
            log.info("STT %.3fs: %s", time.perf_counter() - t0, (user_text or "")[:80])
            if not (user_text and user_text.strip()):
                await self._send_tts(ws, "कृपया दोबारा बोलें।")
                return
            history = self.call_history.get(self.call_sid or "", [])[-4:]
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(history)
            messages.append({"role": "user", "content": user_text.strip()})
            if self.call_sid:
                self.call_history.setdefault(self.call_sid, []).append(
                    {"role": "user", "content": user_text.strip()}
                )
            full_reply = ""
            # Progressive TTS: send each sentence as soon as ready (reference-repo style → ~2–3s to first audio)
            async for sentence in self._stream_llm_sentences(messages):
                if not sentence.strip():
                    continue
                full_reply += sentence.strip() + " "
                if not await self._send_tts_safe(ws, sentence.strip()):
                    break  # WS closed (call ended)
            full_reply = full_reply.strip()
            if self.call_sid and full_reply:
                self.call_history.setdefault(self.call_sid, []).append(
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

    async def _stream_llm_sentences(self, messages: list):
        """Stream LLM; yield only on sentence end (।.?!). Full sentences = natural speech, first audio in ~2–3s."""
        stream = await self.groq_client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            messages=messages,
            max_tokens=128,
            temperature=0.1,
            stream=True,
            tool_choice="none",
        )
        buffer = ""
        async for chunk in stream:
            delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            buffer += delta
            m = SENTENCE_END.search(buffer)
            if m:
                sentence = buffer[: m.end()].strip()
                buffer = buffer[m.end() :]
                if sentence:
                    yield sentence
            elif len(buffer.strip()) >= MAX_BUFFER_CHARS:
                sentence = buffer.strip()
                buffer = ""
                if sentence:
                    yield sentence
        if buffer.strip():
            yield buffer.strip()

    async def _send_mp3_to_ws(self, ws: WebSocket, mp3_bytes: bytes) -> bool:
        """Send MP3 bytes as mulaw chunks to Twilio. Returns False if WS closed."""
        mulaw = _mp3_to_mulaw_8k(mp3_bytes)
        return await self._send_mulaw_to_ws(ws, mulaw)

    async def _send_mulaw_to_ws(self, ws: WebSocket, mulaw: bytes) -> bool:
        """Send mulaw chunks to Twilio. Returns False if WS closed during send."""
        chunk_size = 160
        for i in range(0, len(mulaw), chunk_size):
            chunk = mulaw[i : i + chunk_size]
            b64 = base64.b64encode(chunk).decode("ascii")
            try:
                await ws.send_text(json.dumps({
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": b64},
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
        """Generate TTS for text, send to WS. Returns False if WS closed (call ended)."""
        mp3_bytes = await synthesise_text_bytes(text)
        if mp3_bytes:
            return await self._send_mp3_to_ws(ws, mp3_bytes)
        filename = await synthesise_text(text)
        filepath = os.path.join(AUDIO_DIR, filename)
        try:
            with open(filepath, "rb") as f:
                mp3_bytes = f.read()
            return await self._send_mp3_to_ws(ws, mp3_bytes)
        finally:
            try:
                os.remove(filepath)
            except OSError:
                pass

    async def _send_tts(self, ws: WebSocket, text: str):
        """Generate TTS and send to WS. Swallows errors if WS already closed."""
        await self._send_tts_safe(ws, text)


async def handle_media_stream(
    websocket: WebSocket,
    system_prompt: str,
    call_history: dict,
    groq_client: AsyncGroq,
):
    handler = MediaStreamHandler(system_prompt, call_history, groq_client)
    await handler.handle_connection(websocket)
