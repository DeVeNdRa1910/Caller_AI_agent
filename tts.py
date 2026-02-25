"""
tts.py — Persistent WebSocket TTS + streaming LLM pipeline.

KEY OPTIMISATION: One persistent WS connection reused across ALL turns.
TCP+TLS handshake (~1-2s) happens ONCE at startup, not per-turn.

PIPELINE: LLM tokens stream in → sentence boundaries detected → WS chunks sent
          → audio chunks received CONCURRENTLY → written to file as they arrive.

Sarvam bulbul:v2 WS protocol:
  URL:  wss://api.sarvam.ai/text-to-speech/ws
        ?model=bulbul:v2&send_completion_event=true
  Header: Api-Subscription-Key: <key>

  1. config → {"type":"config","data":{"target_language_code","speaker","pace","min_buffer_size"}}
  2. text   → {"type":"text","data":{"text":"…"}}  (repeat at sentence boundaries)
  3. flush  → {"type":"flush"}
  ← audio  ← {"type":"audio","data":{"audio":"<b64 mp3>"}}  (multiple)
  ← event  ← {"type":"event",…}  (completion — TTS done)
  keepalive: {"type":"ping"} every PING_INTERVAL seconds
"""

import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid

import httpx
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

SARVAM_API_KEY = (os.getenv("SARVAM_API_KEY") or "").strip()
TTS_REST_URL   = "https://api.sarvam.ai/text-to-speech"
TTS_WS_URL     = (
    "wss://api.sarvam.ai/text-to-speech/ws"
    "?model=bulbul:v2&send_completion_event=true"
)

AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_files")
os.makedirs(AUDIO_DIR, exist_ok=True)

DEFAULT_SPEAKER  = "anushka"
DEFAULT_LANGUAGE = "hi-IN"
MIN_BUFFER_SIZE  = 25
PING_INTERVAL    = 20

# Flush to TTS at sentence boundaries — shorter = lower latency first audio
# Added comma+space as a soft flush boundary for long Hindi sentences
_SENTENCE_END = re.compile(r"[।.?!\n]")
# Secondary boundary: flush on comma if buffer is getting long (avoids waiting for full sentence)
_SOFT_FLUSH_LEN = 60   # chars — flush mid-sentence if no hard boundary yet


def _ws_is_open(ws) -> bool:
    if ws is None:
        return False
    try:
        from websockets.protocol import State
        return ws.state == State.OPEN
    except (ImportError, AttributeError):
        pass
    try:
        from websockets.connection import OPEN
        return ws.state == OPEN
    except (ImportError, AttributeError):
        pass
    try:
        return ws.state.name == "OPEN"
    except AttributeError:
        pass
    return getattr(ws, 'open', False)


# ── Persistent WS connection ───────────────────────────────────────────────────

class _WsConn:
    """
    Single persistent Sarvam WS connection for one (language, speaker) pair.
    Reused across all turns — no per-turn handshake.
    """

    def __init__(self, language: str, speaker: str):
        self.language   = language
        self.speaker    = speaker
        self._ws        = None
        self._lock      = asyncio.Lock()
        self._ping_task = None

    # ── Public ─────────────────────────────────────────────────────────────────

    async def synthesise(self, groq_stream) -> tuple[str, str]:
        """Stream LLM tokens → TTS → MP3. Returns (filename, full_text)."""
        async with asyncio.timeout(30):
            async with self._lock:
                return await self._do_synthesise(groq_stream)

    async def close(self):
        if self._ping_task:
            self._ping_task.cancel()
            self._ping_task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    # ── Internal ───────────────────────────────────────────────────────────────

    async def _connect(self):
        t0 = time.perf_counter()
        headers  = {"Api-Subscription-Key": SARVAM_API_KEY}
        self._ws = await websockets.connect(
            TTS_WS_URL,
            additional_headers=headers,
            open_timeout=10,
            close_timeout=5,
            ping_interval=None,
        )
        log.info("WS connected in %.2fs", time.perf_counter() - t0)

        await self._ws.send(json.dumps({
            "type": "config",
            "data": {
                "target_language_code": self.language,
                "speaker":              self.speaker,
                "pace":                 1.05,              # slightly faster speech = less audio time
                "min_buffer_size":      MIN_BUFFER_SIZE,
                "max_chunk_length":     200,
                "output_audio_codec":   "mp3",
                "output_audio_bitrate": "64k",
            }
        }))
        log.info("WS config sent (lang=%s speaker=%s)", self.language, self.speaker)

        if self._ping_task:
            self._ping_task.cancel()
        self._ping_task = asyncio.create_task(self._keepalive())

    async def _ensure_connected(self):
        if not _ws_is_open(self._ws):
            log.info("WS not open — reconnecting…")
            if self._ws:
                try:
                    await self._ws.close()
                except Exception:
                    pass
                self._ws = None
            await self._connect()

    async def _keepalive(self):
        while True:
            await asyncio.sleep(PING_INTERVAL)
            if _ws_is_open(self._ws):
                try:
                    await self._ws.send(json.dumps({"type": "ping"}))
                    log.debug("WS ping sent")
                except Exception as e:
                    log.warning("WS ping failed: %s — will reconnect on next turn", e)
                    self._ws = None
                    break

    async def _do_synthesise(self, groq_stream) -> tuple[str, str]:
        """
        TRUE parallel pipeline:
          - _send_tokens coroutine: reads LLM stream, sends to WS at sentence boundaries
          - _recv_audio coroutine:  receives audio chunks as they arrive
          Both run concurrently via asyncio.gather — TTS starts before LLM finishes.
        """
        t_total      = time.perf_counter()
        audio_chunks : list[bytes] = []
        full_text    = ""

        for attempt in range(2):
            audio_chunks.clear()
            try:
                await self._ensure_connected()

                # ── Coroutine 1: LLM tokens → WS sends ────────────────────────
                async def _send_tokens():
                    nonlocal full_text
                    buf         = ""
                    first_token = True

                    async for chunk in groq_stream:
                        token = chunk.choices[0].delta.content or ""
                        if not token:
                            continue
                        full_text += token
                        buf       += token

                        if first_token:
                            log.info("⏱  LLM first token: %.3fs", time.perf_counter() - t_total)
                            first_token = False

                        # Flush at hard sentence boundary OR soft buffer length
                        should_flush = (
                            _SENTENCE_END.search(buf)
                            or len(buf) >= _SOFT_FLUSH_LEN
                        )
                        if should_flush and buf.strip():
                            await self._ws.send(json.dumps({
                                "type": "text",
                                "data": {"text": buf.strip()}
                            }))
                            log.debug("WS chunk sent (%d chars): %s", len(buf), buf[:60])
                            buf = ""

                    # Send any remaining text
                    if buf.strip():
                        await self._ws.send(json.dumps({
                            "type": "text",
                            "data": {"text": buf.strip()}
                        }))

                    # Signal TTS that we're done sending
                    await self._ws.send(json.dumps({"type": "flush"}))
                    log.info("⏱  LLM done + WS flush: %.3fs", time.perf_counter() - t_total)

                # ── Coroutine 2: WS receives audio chunks ─────────────────────
                async def _recv_audio():
                    t_first = None
                    async for msg in self._ws:
                        data     = json.loads(msg)
                        msg_type = data.get("type")

                        if msg_type == "audio":
                            b64 = (data.get("data") or {}).get("audio") or ""
                            if b64:
                                if t_first is None:
                                    t_first = time.perf_counter()
                                    log.info("⏱  TTS first audio chunk: %.3fs", t_first - t_total)
                                audio_chunks.append(base64.b64decode(b64))

                        elif msg_type == "event":
                            log.info("⏱  TTS complete event: %.3fs", time.perf_counter() - t_total)
                            break   # <-- exits the loop, allowing gather to finish

                        elif msg_type == "error":
                            raise RuntimeError(f"Sarvam WS error: {data}")

                        elif msg_type == "pong":
                            log.debug("WS pong received")

                # Run both concurrently — _recv_audio exits when it sees "event"
                # _send_tokens feeds in LLM tokens as they arrive
                await asyncio.gather(_send_tokens(), _recv_audio())
                break   # success

            except (ConnectionClosed, WebSocketException, OSError, RuntimeError) as e:
                log.warning("WS attempt %d failed: %s", attempt + 1, e)
                self._ws = None
                if attempt == 1:
                    log.warning("WS failed twice → REST fallback")
                    if full_text.strip():
                        fname = await generate_tts_async(full_text.strip(), self.language, self.speaker)
                        return fname, full_text.strip()
                    raise

        full_text = full_text.strip() or "कृपया दोबारा बोलें।"

        if not audio_chunks:
            log.warning("WS: no audio chunks — REST fallback")
            fname = await generate_tts_async(full_text, self.language, self.speaker)
            return fname, full_text

        combined  = b"".join(audio_chunks)
        filename  = f"{uuid.uuid4()}.mp3"
        filepath  = os.path.join(AUDIO_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(combined)
        log.info("⏱  TOTAL pipeline: %.3fs | %d bytes | text: %s",
                 time.perf_counter() - t_total, len(combined), full_text[:80])
        return filename, full_text


# ── Connection pool ────────────────────────────────────────────────────────────

_pool      : dict[tuple[str, str], _WsConn] = {}
_pool_lock = asyncio.Lock()


async def _get_conn(language: str, speaker: str) -> _WsConn:
    key = (language, speaker)
    async with _pool_lock:
        if key not in _pool:
            conn = _WsConn(language, speaker)
            await conn._connect()
            _pool[key] = conn
        return _pool[key]


async def warmup_ws(language: str = DEFAULT_LANGUAGE, speaker: str = DEFAULT_SPEAKER):
    """Pre-open WS at startup so first call turn pays zero handshake cost."""
    log.info("Warming up WS TTS (%s / %s)…", language, speaker)
    await _get_conn(language, speaker)
    log.info("WS TTS warm-up done.")


# ── Public API ─────────────────────────────────────────────────────────────────

async def llm_to_tts_stream(
    groq_stream,
    language: str | None = None,
    speaker:  str | None = None,
) -> tuple[str, str]:
    """Stream Groq tokens → persistent Sarvam WS TTS → MP3. Returns (filename, full_text)."""
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY not set")
    language = language or DEFAULT_LANGUAGE
    speaker  = speaker  or DEFAULT_SPEAKER
    conn = await _get_conn(language, speaker)
    return await conn.synthesise(groq_stream)


async def close_all_ws():
    async with _pool_lock:
        for conn in _pool.values():
            await conn.close()
        _pool.clear()


# ── REST client (fallback + one-shot audio) ────────────────────────────────────

_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=10),
            timeout=httpx.Timeout(15.0),
        )
    return _http_client


async def close_http_client():
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
    await close_all_ws()


def _is_mostly_hindi(text: str) -> bool:
    letters = [c for c in text if c.strip()]
    if not letters:
        return False
    dev = sum(1 for c in letters if "\u0900" <= c <= "\u097F")
    return (dev / len(letters)) > 0.25


async def generate_tts_async(
    text: str,
    language: str | None = None,
    speaker:  str | None = None,
    model:    str | None = None,
) -> str:
    """REST TTS — for static one-shot text (opening message). Returns filename."""
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY not set")
    language = language or ("hi-IN" if _is_mostly_hindi(text) else DEFAULT_LANGUAGE)
    speaker  = speaker  or DEFAULT_SPEAKER
    model    = model    or "bulbul:v2"

    payload = {
        "text":                 text,
        "target_language_code": language,
        "speaker":              speaker,
        "model":                model,
        "output_audio_codec":   "mp3",
        "pace":                 1.05,
    }
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type":         "application/json",
    }

    t0   = time.perf_counter()
    resp = await _get_http_client().post(TTS_REST_URL, json=payload, headers=headers)
    if not resp.is_success:
        try:    err = resp.json()
        except: err = resp.text or f"HTTP {resp.status_code}"
        if resp.status_code == 403:
            raise ValueError(f"Sarvam 403 — invalid/expired key. {err}")
        resp.raise_for_status()

    audios = resp.json().get("audios") or []
    if not audios:
        raise ValueError("Sarvam REST TTS returned no audio")

    audio_bytes = base64.b64decode(audios[0])
    filename    = f"{uuid.uuid4()}.mp3"
    with open(os.path.join(AUDIO_DIR, filename), "wb") as f:
        f.write(audio_bytes)
    log.info("REST TTS done in %.2fs: %s (%d bytes)",
             time.perf_counter() - t0, filename, len(audio_bytes))
    return filename


def generate_tts(text: str, language: str | None = None, speaker: str | None = None) -> str:
    return asyncio.run(generate_tts_async(text, language, speaker))