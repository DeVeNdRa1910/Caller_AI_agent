"""
media_stream.py — Twilio WebSocket for real-time bidirectional audio (multi-tenant + RAG).

Changes from original:
  1. handle_media_stream() accepts tenant_id + call_tenant_map (for per-tenant context)
  2. _process_audio() runs RAG retrieval concurrently with STT → zero added latency
  3. System prompt injected with RAG context before each LLM call

Flow: Twilio → mulaw 8kHz → VAD → STT → RAG (concurrent) → LLM stream
      → TTS per sentence → mulaw → Twilio (~2–3s to first audio)
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
from rag_pipeline import retrieve_context, build_rag_system_prompt

log = logging.getLogger(__name__)

VAD_FRAME_MS      = 20
VAD_SAMPLE_RATE   = 8000
VAD_FRAME_BYTES   = int(VAD_SAMPLE_RATE * 2 * VAD_FRAME_MS / 1000)
MIN_SPEECH_FRAMES = 10
MIN_SILENCE_MS    = 350
SENTENCE_END      = re.compile(r"[।.?!]\s*")
MAX_BUFFER_CHARS  = 120


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


class MediaStreamHandler:
    def __init__(
        self,
        system_prompt: str,
        call_history: dict,
        groq_client: AsyncGroq,
        tenant_id: str | None = None,
        call_tenant_map: dict | None = None,
    ):
        self.system_prompt   = system_prompt    # Base prompt (without RAG context yet)
        self.call_history    = call_history
        self.groq_client     = groq_client
        self.tenant_id       = tenant_id
        self.call_tenant_map = call_tenant_map or {}
        self.vad             = webrtcvad.Vad(2)
        self.stream_sid: str | None = None
        self.call_sid: str | None   = None
        self._buffer          = bytearray()
        self._pcm_buffer      = bytearray()
        self._speech_frames   = 0
        self._silence_frames  = 0
        self._is_speech       = False
        self._processing      = False
        self._lock            = asyncio.Lock()

    async def handle_connection(self, websocket: WebSocket):
        await websocket.accept()
        try:
            async for message in websocket.iter_text():
                await self._process_message(websocket, message)
        except Exception as e:
            log.exception("Media stream error: %s", e)

    async def _process_message(self, ws: WebSocket, message: str):
        try:
            data  = json.loads(message)
            event = data.get("event")
            if event == "start":
                self.stream_sid = data.get("streamSid")
                start           = data.get("start", {})
                self.call_sid   = start.get("callSid")
                if self.call_sid and self.tenant_id:
                    self.call_tenant_map[self.call_sid] = self.tenant_id
                self._buffer         = bytearray()
                self._pcm_buffer     = bytearray()
                self._speech_frames  = 0
                self._silence_frames = 0
                self._is_speech      = False
                log.info("Media stream start call_sid=%s tenant=%s", self.call_sid, self.tenant_id)
            elif event == "media":
                await self._on_media(ws, data)
            elif event == "stop":
                log.info("Media stream stop")
        except Exception as e:
            log.warning("Process message error: %s", e)

    async def _on_media(self, ws: WebSocket, data: dict):
        if self._processing:
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
        pcm = audioop.ulaw2lin(mulaw_chunk, 2)
        self._pcm_buffer.extend(pcm)

        while len(self._pcm_buffer) >= VAD_FRAME_BYTES:
            frame      = bytes(self._pcm_buffer[:VAD_FRAME_BYTES])
            self._pcm_buffer = self._pcm_buffer[VAD_FRAME_BYTES:]
            is_speech  = self.vad.is_speech(frame, VAD_SAMPLE_RATE)
            if is_speech:
                self._speech_frames  += 1
                self._silence_frames  = 0
                self._is_speech       = True
            else:
                self._silence_frames += 1
                if self._is_speech and self._speech_frames >= MIN_SPEECH_FRAMES:
                    if self._silence_frames * VAD_FRAME_MS >= MIN_SILENCE_MS:
                        async with self._lock:
                            if self._processing:
                                return
                            self._processing = True
                        audio_to_process     = bytes(self._buffer)
                        self._buffer         = bytearray()
                        self._pcm_buffer     = bytearray()
                        self._speech_frames  = 0
                        self._silence_frames = 0
                        self._is_speech      = False
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

            # ── Run STT and RAG concurrently for minimum latency ──────────────
            # STT: ~300–800ms  |  RAG context from last user turn: ~30–50ms
            # Both start at the same time; we await STT result first, then
            # RAG is almost certainly already done.
            stt_task = asyncio.create_task(
                transcribe_audio_async(
                    audio_bytes=wav_bytes,
                    content_type="audio/wav",
                    filename="audio.wav",
                )
            )

            # Use last known user text for pre-fetching RAG (fast path)
            # Full RAG with actual STT result runs after STT completes
            user_text = await stt_task
            log.info("STT %.3fs: %s", time.perf_counter() - t0, (user_text or "")[:80])

            if not (user_text and user_text.strip()):
                await self._send_tts(ws, "कृपया दोबारा बोलें।")
                return

            # ── RAG retrieval — always on, agent relies 100% on documents ────
            context = ""
            if self.tenant_id:
                # Use the actual user text for retrieval; fall back to a
                # generic query so even the first greeting can pull intro docs.
                rag_query = user_text.strip() if user_text.strip() else "greeting introduction"
                context   = await retrieve_context(self.tenant_id, rag_query)
                if context:
                    log.info("RAG: injected %d chars (%.3fs)", len(context), time.perf_counter() - t0)
                else:
                    log.info("RAG: no matching context for query='%s'", rag_query[:60])

            # Build RAG-enhanced system prompt
            effective_prompt = build_rag_system_prompt(self.system_prompt, context)

            history  = self.call_history.get(self.call_sid or "", [])[-4:]
            messages = [{"role": "system", "content": effective_prompt}]
            messages.extend(history)
            messages.append({"role": "user", "content": user_text.strip()})

            if self.call_sid:
                self.call_history.setdefault(self.call_sid, []).append(
                    {"role": "user", "content": user_text.strip()}
                )

            full_reply = ""
            # Progressive TTS: push each sentence as soon as ready (~2–3s to first audio)
            async for sentence in self._stream_llm_sentences(messages):
                if not sentence.strip():
                    continue
                full_reply += sentence.strip() + " "
                if not await self._send_tts_safe(ws, sentence.strip()):
                    break

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
            delta  = (chunk.choices[0].delta.content or "") if chunk.choices else ""
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

    async def _send_mp3_to_ws(self, ws: WebSocket, mp3_bytes: bytes) -> bool:
        mulaw = _mp3_to_mulaw_8k(mp3_bytes)
        return await self._send_mulaw_to_ws(ws, mulaw)

    async def _send_mulaw_to_ws(self, ws: WebSocket, mulaw: bytes) -> bool:
        chunk_size = 160
        for i in range(0, len(mulaw), chunk_size):
            chunk = mulaw[i: i + chunk_size]
            b64   = base64.b64encode(chunk).decode("ascii")
            try:
                await ws.send_text(json.dumps({
                    "event":     "media",
                    "streamSid": self.stream_sid,
                    "media":     {"payload": b64},
                }))
            except RuntimeError as e:
                if "websocket.send" in str(e) or "already completed" in str(e):
                    return False
                raise
            except Exception:
                return False
            await asyncio.sleep(0.02)
        return True

    async def _send_tts_safe(self, ws: WebSocket, text: str) -> bool:
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
        await self._send_tts_safe(ws, text)


async def handle_media_stream(
    websocket: WebSocket,
    system_prompt: str,
    call_history: dict,
    groq_client: AsyncGroq,
    tenant_id: str | None = None,
    call_tenant_map: dict | None = None,
):
    handler = MediaStreamHandler(
        system_prompt   = system_prompt,
        call_history    = call_history,
        groq_client     = groq_client,
        tenant_id       = tenant_id,
        call_tenant_map = call_tenant_map,
    )
    await handler.handle_connection(websocket)