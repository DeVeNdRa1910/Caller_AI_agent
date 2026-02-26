"""
tts.py — Persistent WebSocket TTS with keepalive pings.

KEY CHANGE: Sarvam WS supports a "ping" message type to keep the connection
alive (1-minute idle timeout). We now maintain ONE persistent connection per
language/speaker pair across ALL turns, eliminating the ~300ms reconnect cost
on every turn.

ARCHITECTURE:
  - _PersistentWsConn keeps one WS open indefinitely with a background keepalive task
  - Per turn: send config (once, first time) → send text chunks → flush → recv audio
  - Since connection is already open, per-turn overhead is near-zero
  - REST fallback if WS fails, with instant retry on next turn

Sarvam bulbul:v3 WS protocol:
  URL:  wss://api.sarvam.ai/text-to-speech/ws
        ?model=bulbul:v3&send_completion_event=true
  Header: Api-Subscription-Key: <key>
  1. config    → {"type":"config","data":{"target_language_code","speaker","pace"}}
  2. text      → {"type":"text","data":{"text":"…"}}
  3. flush     → {"type":"flush"}
  4. ping      → {"type":"ping"}  ← keepalive, send every 30s of idle
  ← audio     ← {"type":"audio","data":{"audio":"<b64 mp3>"}}
  ← event     ← {"type":"event",...}  ← signals TTS done, connection stays open
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

TTS_MODEL   = "bulbul:v3"
TTS_SPEAKER = "simran"
TTS_WS_URL  = f"wss://api.sarvam.ai/text-to-speech/ws?model={TTS_MODEL}&send_completion_event=true"

AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_files")
os.makedirs(AUDIO_DIR, exist_ok=True)

DEFAULT_SPEAKER  = TTS_SPEAKER
DEFAULT_LANGUAGE = "hi-IN"

_SENTENCE_END   = re.compile(r"[।.?!\n]")
_SOFT_FLUSH_LEN = 50   # chars: flush mid-sentence to start TTS early
_PING_INTERVAL  = 30   # seconds between keepalive pings


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

class _PersistentWsConn:
    """
    Single long-lived Sarvam WS connection with keepalive pings.

    Lifecycle:
      - connect() once (done at startup by warmup_ws)
      - keepalive task sends {"type":"ping"} every 30s to prevent idle disconnect
      - synthesise() sends text+flush, receives audio, connection stays open
      - If connection dies mid-turn → REST fallback, reconnect on next call
    """

    def __init__(self, language: str, speaker: str):
        self.language       = language
        self.speaker        = speaker
        self._ws            = None
        self._lock          = asyncio.Lock()
        self._keepalive_task: asyncio.Task | None = None
        self._config_sent   = False

    # ── Public ──────────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        """Establish WS and start keepalive. Safe to call multiple times."""
        if _ws_is_open(self._ws):
            return True
        return await self._connect()

    async def synthesise(self, text: str) -> tuple[str, str]:
        """
        Synthesise a pre-known text string (no LLM stream).
        Returns (filename, text).
        """
        async with asyncio.timeout(20):
            async with self._lock:
                return await self._do_synthesise_text(text)

    async def synthesise_stream(self, groq_stream) -> tuple[str, str]:
        """
        Synthesise from a live LLM stream (concurrent LLM drain + WS send).
        Returns (filename, full_text).
        """
        async with asyncio.timeout(25):
            async with self._lock:
                return await self._do_synthesise_stream(groq_stream)

    async def close(self):
        if self._keepalive_task:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
            self._keepalive_task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._config_sent = False

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _connect(self) -> bool:
        t0 = time.perf_counter()
        try:
            headers  = {"Api-Subscription-Key": SARVAM_API_KEY}
            self._ws = await websockets.connect(
                TTS_WS_URL,
                additional_headers=headers,
                open_timeout=8,
                close_timeout=3,
                ping_interval=None,   # we handle pings manually
            )
            log.info("WS connected in %.3fs", time.perf_counter() - t0)
            self._config_sent = False

            # Send config immediately after connect
            await self._send_config()

            # Start keepalive in background
            if self._keepalive_task is None or self._keepalive_task.done():
                self._keepalive_task = asyncio.create_task(self._keepalive_loop())

            return True
        except Exception as e:
            log.warning("WS connect failed: %s", e)
            self._ws            = None
            self._config_sent   = False
            return False

    async def _send_config(self):
        """Send config frame (once per connection)."""
        if not _ws_is_open(self._ws):
            return
        await self._ws.send(json.dumps({
            "type": "config",
            "data": {
                "target_language_code": self.language,
                "speaker":              self.speaker,
                "pace":                 1.0,
            }
        }))
        self._config_sent = True
        log.info("WS config sent (lang=%s speaker=%s)", self.language, self.speaker)

    async def _keepalive_loop(self):
        """Send ping every _PING_INTERVAL seconds to prevent idle disconnect."""
        try:
            while True:
                await asyncio.sleep(_PING_INTERVAL)
                if not _ws_is_open(self._ws):
                    log.info("Keepalive: WS closed, stopping ping task")
                    return
                try:
                    await self._ws.send(json.dumps({"type": "ping"}))
                    log.debug("Keepalive ping sent")
                except Exception as e:
                    log.warning("Keepalive ping failed: %s — will reconnect on next turn", e)
                    self._ws          = None
                    self._config_sent = False
                    return
        except asyncio.CancelledError:
            pass

    async def _ensure_ws(self) -> bool:
        """Reconnect if needed before a turn. Returns True if WS is ready."""
        if _ws_is_open(self._ws) and self._config_sent:
            return True
        log.info("WS not ready — reconnecting…")
        return await self._connect()

    async def _do_synthesise_text(self, text: str) -> tuple[str, str]:
        """Synthesise from a plain string (cache pre-render path)."""
        t0 = time.perf_counter()
        ws_ok = await self._ensure_ws()
        if not ws_ok:
            log.info("WS unavailable — REST fallback for text synth")
            fname = await generate_tts_async(text, self.language, self.speaker)
            return fname, text

        try:
            audio_chunks: list[bytes] = []
            await self._ws.send(json.dumps({"type": "text", "data": {"text": text.strip()}}))
            await self._ws.send(json.dumps({"type": "flush"}))
            log.info("⏱  WS flush (text): %.3fs", time.perf_counter() - t0)

            async for msg in self._ws:
                data     = json.loads(msg)
                msg_type = data.get("type")
                if msg_type == "audio":
                    b64 = (data.get("data") or {}).get("audio") or ""
                    if b64:
                        audio_chunks.append(base64.b64decode(b64))
                elif msg_type == "event":
                    log.info("⏱  TTS complete (text): %.3fs", time.perf_counter() - t0)
                    break
                elif msg_type == "error":
                    raise RuntimeError(f"Sarvam WS error: {data}")

            if audio_chunks:
                return _save_audio(audio_chunks, text, t0)
            log.warning("WS: no audio chunks — REST fallback")

        except (ConnectionClosed, WebSocketException, OSError, RuntimeError) as e:
            log.warning("WS error during text synth: %s — REST fallback", e)
            self._ws          = None
            self._config_sent = False

        fname = await generate_tts_async(text, self.language, self.speaker)
        return fname, text

    async def _do_synthesise_stream(self, groq_stream) -> tuple[str, str]:
        """Synthesise from live LLM stream: LLM drain + WS send run concurrently."""
        t_total      = time.perf_counter()
        audio_chunks : list[bytes] = []
        full_text    = ""
        llm_buffer   : list[str]   = []
        llm_done_evt = asyncio.Event()

        async def _drain_llm():
            nonlocal full_text
            first = True
            async for chunk in groq_stream:
                token = chunk.choices[0].delta.content or ""
                if not token:
                    continue
                full_text += token
                llm_buffer.append(token)
                if first:
                    log.info("⏱  LLM first token: %.3fs", time.perf_counter() - t_total)
                    first = False
            log.info("⏱  LLM done: %.3fs | %s", time.perf_counter() - t_total, full_text[:60])
            llm_done_evt.set()

        llm_task = asyncio.create_task(_drain_llm())

        # Check WS health — fast if already connected (~0ms), slow if need reconnect (~300ms)
        # Both run somewhat concurrently: LLM draining while we check/reconnect WS
        ws_ok = await self._ensure_ws()
        log.info("⏱  WS ready check: %.3fs (ok=%s)", time.perf_counter() - t_total, ws_ok)

        if ws_ok:
            try:
                async def _send_from_buffer():
                    buf = ""
                    while True:
                        while llm_buffer:
                            buf += llm_buffer.pop(0)
                        if buf.strip() and (_SENTENCE_END.search(buf) or len(buf) >= _SOFT_FLUSH_LEN):
                            await self._ws.send(json.dumps({"type": "text", "data": {"text": buf.strip()}}))
                            log.debug("WS chunk (%dc): %s", len(buf), buf[:60])
                            buf = ""
                        if llm_done_evt.is_set() and not llm_buffer:
                            break
                        await asyncio.sleep(0.005)
                    if buf.strip():
                        await self._ws.send(json.dumps({"type": "text", "data": {"text": buf.strip()}}))
                    await self._ws.send(json.dumps({"type": "flush"}))
                    log.info("⏱  WS flush: %.3fs", time.perf_counter() - t_total)

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
                                    log.info("⏱  TTS first audio: %.3fs", t_first - t_total)
                                audio_chunks.append(base64.b64decode(b64))
                        elif msg_type == "event":
                            log.info("⏱  TTS complete: %.3fs", time.perf_counter() - t_total)
                            break
                        elif msg_type == "error":
                            raise RuntimeError(f"Sarvam WS error: {data}")

                await asyncio.gather(llm_task, _send_from_buffer(), _recv_audio())
                full_text = full_text.strip() or "कृपया दोबारा बोलें।"
                if audio_chunks:
                    return _save_audio(audio_chunks, full_text, t_total)
                log.warning("WS: no audio — REST fallback")

            except (ConnectionClosed, WebSocketException, OSError, RuntimeError) as e:
                log.warning("WS error: %s — REST fallback", e)
                self._ws          = None
                self._config_sent = False

        # REST fallback
        if not llm_task.done():
            await llm_task
        full_text = full_text.strip() or "कृपया दोबारा बोलें।"
        log.info("⏱  REST text ready: %.3fs", time.perf_counter() - t_total)
        fname = await generate_tts_async(full_text, self.language, self.speaker)
        log.info("⏱  TOTAL (REST fallback): %.3fs", time.perf_counter() - t_total)
        return fname, full_text


def _save_audio(chunks: list[bytes], full_text: str, t0: float) -> tuple[str, str]:
    combined = b"".join(chunks)
    filename = f"{uuid.uuid4()}.mp3"
    with open(os.path.join(AUDIO_DIR, filename), "wb") as f:
        f.write(combined)
    log.info("⏱  TOTAL (WS path): %.3fs | %d bytes | %s",
             time.perf_counter() - t0, len(combined), full_text[:80])
    return filename, full_text


# ── Connection pool ────────────────────────────────────────────────────────────

_pool      : dict[tuple[str, str], _PersistentWsConn] = {}
_pool_lock = asyncio.Lock()


async def _get_conn(language: str, speaker: str) -> _PersistentWsConn:
    key = (language, speaker)
    async with _pool_lock:
        if key not in _pool:
            _pool[key] = _PersistentWsConn(language, speaker)
        return _pool[key]


async def warmup_ws(language: str = DEFAULT_LANGUAGE, speaker: str = DEFAULT_SPEAKER):
    """
    Warm up: establish the persistent WS connection at startup.
    After this returns, all turns get near-zero WS connect overhead.
    """
    log.info("Warming up persistent WS TTS (%s / %s)…", language, speaker)
    conn = await _get_conn(language, speaker)
    ok   = await conn.connect()
    if ok:
        log.info("✅ Persistent WS TTS ready — keepalive active.")
    else:
        log.warning("⚠️  WS warm-up failed — will use REST TTS until reconnect.")


async def close_all_ws():
    async with _pool_lock:
        for conn in _pool.values():
            await conn.close()
        _pool.clear()


# ── Public API ─────────────────────────────────────────────────────────────────

async def llm_to_tts_stream(
    groq_stream,
    language: str | None = None,
    speaker:  str | None = None,
) -> tuple[str, str]:
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY not set")
    language = language or DEFAULT_LANGUAGE
    speaker  = speaker  or DEFAULT_SPEAKER
    conn     = await _get_conn(language, speaker)
    return await conn.synthesise_stream(groq_stream)


async def synthesise_text(
    text: str,
    language: str | None = None,
    speaker:  str | None = None,
) -> str:
    """
    Synthesise known text via persistent WS (fastest path: no LLM needed).
    Used by the pre-render cache. Returns filename.
    """
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY not set")
    language = language or DEFAULT_LANGUAGE
    speaker  = speaker  or DEFAULT_SPEAKER
    conn     = await _get_conn(language, speaker)
    filename, _ = await conn.synthesise(text)
    return filename


async def close_http_client():
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
    await close_all_ws()


# ── REST client (fallback) ─────────────────────────────────────────────────────

_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=10),
            timeout=httpx.Timeout(15.0),
        )
    return _http_client


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
    """REST TTS — reliable fallback. Returns filename."""
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY not set")
    language = language or ("hi-IN" if _is_mostly_hindi(text) else DEFAULT_LANGUAGE)
    speaker  = speaker  or DEFAULT_SPEAKER
    model    = model    or TTS_MODEL

    payload = {
        "text":                 text,
        "target_language_code": language,
        "speaker":              speaker,
        "model":                model,
        "pace":                 1.0,
        "output_audio_codec":   "mp3",
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
    log.info("REST TTS done in %.3fs: %s (%d bytes)",
             time.perf_counter() - t0, filename, len(audio_bytes))
    return filename


def generate_tts(text: str, language: str | None = None, speaker: str | None = None) -> str:
    return asyncio.run(generate_tts_async(text, language, speaker))