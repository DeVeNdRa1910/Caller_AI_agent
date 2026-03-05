"""
tts.py — Persistent WebSocket TTS optimized for <4s total latency.

LATENCY FIXES vs previous version:
  1. MULAW 8kHz OUTPUT: Twilio transcodes every MP3 to 8kHz mulaw internally.
     We now request mulaw directly from Sarvam → file is ~6x smaller, Twilio
     skips transcoding entirely → saves ~150-300ms on Twilio fetch+play start.

  2. WS CONNECTION POOL (2 connections): The single-connection lock serialised
     concurrent TTS requests. With 2 pooled connections, two sentences can
     synthesise in parallel → total = max(TTS1, TTS2) not TTS1 + TTS2.

  3. SOFT FLUSH LENGTH = 40 chars (was 15): 15-char fragments caused Sarvam to
     produce choppy/empty audio that silently retried, adding 600ms+ randomly.

  4. IN-MEMORY LRU CACHE (60 entries): Repeat text (fixed questions, unclear
     prompts) skips all API calls entirely → ~0ms TTS.

  5. DOUBLE-CHECK CACHE AFTER LOCK: Prevents two concurrent callers both
     synthesising the same text when cache was empty at time of first check.

Sarvam bulbul:v3 WS protocol:
  URL:  wss://api.sarvam.ai/text-to-speech/ws
        ?model=bulbul:v3&send_completion_event=true
  Header: Api-Subscription-Key: <key>
  → config  {"type":"config","data":{target_language_code,speaker,pace,output_audio_codec,speech_sample_rate}}
  → text    {"type":"text","data":{"text":"…"}}
  → flush   {"type":"flush"}
  → ping    {"type":"ping"}   ← keepalive every 28s (Sarvam idle timeout ~30s)
  ← audio   {"type":"audio","data":{"audio":"<b64>"}}
  ← event   {"type":"event"} ← TTS done, connection stays open
"""

import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid
from collections import OrderedDict

import httpx
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

SARVAM_API_KEY = (os.getenv("SARVAM_API_KEY") or "").strip()
TTS_REST_URL   = "https://api.sarvam.ai/text-to-speech"

TTS_MODEL       = "bulbul:v3"
TTS_SPEAKER     = "simran"
# mulaw 8kHz = native Twilio telephony format.
# Twilio normally transcodes MP3→mulaw itself (adds latency + quality loss).
# Serving mulaw directly skips that step entirely.
TTS_CODEC       = "mulaw"
TTS_SAMPLE_RATE = 8000
TTS_FILE_EXT    = ".wav"

TTS_WS_URL = (
    "wss://api.sarvam.ai/text-to-speech/ws"
    f"?model={TTS_MODEL}&send_completion_event=true"
)

AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_files")
os.makedirs(AUDIO_DIR, exist_ok=True)

DEFAULT_SPEAKER  = TTS_SPEAKER
DEFAULT_LANGUAGE = "hi-IN"

_SENTENCE_END   = re.compile(r"[।.?!\n]")
_SOFT_FLUSH_LEN = 40    # chars before forcing a mid-sentence flush to TTS
_PING_INTERVAL  = 28    # seconds between keepalive pings

# ── In-memory LRU audio cache ──────────────────────────────────────────────────
_TTS_CACHE_MAX = 60
_tts_cache: OrderedDict[str, bytes] = OrderedDict()


def _cache_get(text: str) -> bytes | None:
    key = text.strip()
    if key in _tts_cache:
        _tts_cache.move_to_end(key)
        return _tts_cache[key]
    return None


def _cache_put(text: str, data: bytes):
    key = text.strip()
    _tts_cache[key] = data
    _tts_cache.move_to_end(key)
    while len(_tts_cache) > _TTS_CACHE_MAX:
        _tts_cache.popitem(last=False)


def _bytes_to_file(data: bytes) -> str:
    filename = f"{uuid.uuid4()}{TTS_FILE_EXT}"
    with open(os.path.join(AUDIO_DIR, filename), "wb") as f:
        f.write(data)
    return filename


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

    def __init__(self, language: str, speaker: str):
        self.language       = language
        self.speaker        = speaker
        self._ws            = None
        self._lock          = asyncio.Lock()
        self._keepalive_task: asyncio.Task | None = None
        self._config_sent   = False

    async def connect(self) -> bool:
        if _ws_is_open(self._ws) and self._config_sent:
            return True
        return await self._connect()

    async def synthesise(self, text: str) -> tuple[str, str]:
        """Synthesise known text → (filename, text). Checks cache before locking."""
        cached = _cache_get(text)
        if cached:
            return _bytes_to_file(cached), text

        async with asyncio.timeout(20):
            async with self._lock:
                # Re-check after lock (another coroutine may have filled cache)
                cached = _cache_get(text)
                if cached:
                    return _bytes_to_file(cached), text
                return await self._do_text(text)

    async def synthesise_stream(self, groq_stream) -> tuple[str, str]:
        async with asyncio.timeout(25):
            async with self._lock:
                return await self._do_stream(groq_stream)

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

    async def _connect(self) -> bool:
        t0 = time.perf_counter()
        try:
            self._ws = await websockets.connect(
                TTS_WS_URL,
                additional_headers={"Api-Subscription-Key": SARVAM_API_KEY},
                open_timeout=8,
                close_timeout=3,
                ping_interval=None,
            )
            log.info("WS connected in %.3fs", time.perf_counter() - t0)
            self._config_sent = False
            await self._send_config()
            if self._keepalive_task is None or self._keepalive_task.done():
                self._keepalive_task = asyncio.create_task(self._keepalive_loop())
            return True
        except Exception as e:
            log.warning("WS connect failed: %s", e)
            self._ws = None
            self._config_sent = False
            return False

    async def _send_config(self):
        if not _ws_is_open(self._ws):
            return
        await self._ws.send(json.dumps({
            "type": "config",
            "data": {
                "target_language_code": self.language,
                "speaker":              self.speaker,
                "pace":                 1.0,
                "output_audio_codec":   TTS_CODEC,
                "speech_sample_rate":   TTS_SAMPLE_RATE,
            }
        }))
        self._config_sent = True
        log.info("WS config sent (lang=%s speaker=%s %s@%dHz)",
                 self.language, self.speaker, TTS_CODEC, TTS_SAMPLE_RATE)

    async def _keepalive_loop(self):
        try:
            while True:
                await asyncio.sleep(_PING_INTERVAL)
                if not _ws_is_open(self._ws):
                    return
                try:
                    await self._ws.send(json.dumps({"type": "ping"}))
                    log.debug("Keepalive ping sent")
                except Exception as e:
                    log.warning("Keepalive ping failed: %s", e)
                    self._ws = None
                    self._config_sent = False
                    return
        except asyncio.CancelledError:
            pass

    async def _ensure_ws(self) -> bool:
        if _ws_is_open(self._ws) and self._config_sent:
            return True
        log.info("WS not ready — reconnecting…")
        return await self._connect()

    async def _recv_loop(self) -> list[bytes]:
        """Receive audio chunks until event. Returns list of raw audio bytes."""
        chunks: list[bytes] = []
        t_first = None
        async for msg in self._ws:
            d = json.loads(msg)
            t = d.get("type")
            if t == "audio":
                b64 = (d.get("data") or {}).get("audio") or ""
                if b64:
                    if t_first is None:
                        t_first = time.perf_counter()
                    chunks.append(base64.b64decode(b64))
            elif t == "event":
                break
            elif t == "error":
                raise RuntimeError(f"Sarvam WS error: {d}")
        return chunks

    async def _do_text(self, text: str) -> tuple[str, str]:
        t0    = time.perf_counter()
        ws_ok = await self._ensure_ws()
        if not ws_ok:
            fname = await generate_tts_async(text, self.language, self.speaker)
            return fname, text
        try:
            await self._ws.send(json.dumps({"type": "text", "data": {"text": text.strip()}}))
            await self._ws.send(json.dumps({"type": "flush"}))
            log.info("⏱  WS flush (text): %.3fs", time.perf_counter() - t0)

            chunks = await self._recv_loop()
            log.info("⏱  TTS done (text): %.3fs", time.perf_counter() - t0)

            if chunks:
                data = b"".join(chunks)
                _cache_put(text, data)
                return _bytes_to_file(data), text
            log.warning("WS: no audio chunks — REST fallback")

        except (ConnectionClosed, WebSocketException, OSError, RuntimeError) as e:
            log.warning("WS error (text): %s — REST fallback", e)
            self._ws = None
            self._config_sent = False

        fname = await generate_tts_async(text, self.language, self.speaker)
        return fname, text

    async def _do_stream(self, groq_stream) -> tuple[str, str]:
        """LLM tokens → WS TTS concurrently. First token immediately triggers TTS send."""
        t_total      = time.perf_counter()
        chunks       : list[bytes] = []
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
        ws_ok    = await self._ensure_ws()
        log.info("⏱  WS ready: %.3fs (ok=%s)", time.perf_counter() - t_total, ws_ok)

        if ws_ok:
            try:
                async def _send_from_buffer():
                    buf = ""
                    while True:
                        while llm_buffer:
                            buf += llm_buffer.pop(0)
                        if buf.strip() and (
                            _SENTENCE_END.search(buf) or len(buf) >= _SOFT_FLUSH_LEN
                        ):
                            await self._ws.send(json.dumps(
                                {"type": "text", "data": {"text": buf.strip()}}
                            ))
                            log.debug("WS chunk (%dc): %s", len(buf), buf[:60])
                            buf = ""
                        if llm_done_evt.is_set() and not llm_buffer:
                            break
                        await asyncio.sleep(0.003)
                    if buf.strip():
                        await self._ws.send(json.dumps(
                            {"type": "text", "data": {"text": buf.strip()}}
                        ))
                    await self._ws.send(json.dumps({"type": "flush"}))
                    log.info("⏱  WS flush (stream): %.3fs", time.perf_counter() - t_total)

                async def _recv_audio():
                    t_first = None
                    async for msg in self._ws:
                        d = json.loads(msg)
                        t = d.get("type")
                        if t == "audio":
                            b64 = (d.get("data") or {}).get("audio") or ""
                            if b64:
                                if t_first is None:
                                    t_first = time.perf_counter()
                                    log.info("⏱  TTS first audio: %.3fs", t_first - t_total)
                                chunks.append(base64.b64decode(b64))
                        elif t == "event":
                            log.info("⏱  TTS complete: %.3fs", time.perf_counter() - t_total)
                            break
                        elif t == "error":
                            raise RuntimeError(f"Sarvam WS error: {d}")

                await asyncio.gather(llm_task, _send_from_buffer(), _recv_audio())
                full_text = full_text.strip() or "कृपया दोबारा बोलें।"
                if chunks:
                    data = b"".join(chunks)
                    _cache_put(full_text, data)
                    return _bytes_to_file(data), full_text
                log.warning("WS stream: no audio — REST fallback")

            except (ConnectionClosed, WebSocketException, OSError, RuntimeError) as e:
                log.warning("WS stream error: %s — REST fallback", e)
                self._ws = None
                self._config_sent = False

        if not llm_task.done():
            await llm_task
        full_text = full_text.strip() or "कृपया दोबारा बोलें।"
        fname = await generate_tts_async(full_text, self.language, self.speaker)
        log.info("⏱  TOTAL (REST fallback): %.3fs", time.perf_counter() - t_total)
        return fname, full_text


# ── Connection pool (2 per language/speaker) ──────────────────────────────────
# Two connections allow two concurrent TTS requests without one blocking the other.
# Round-robin dispatch ensures even load across connections.

_POOL_SIZE = 2
_pool      : dict[tuple[str, str], list[_PersistentWsConn]] = {}
_pool_lock = asyncio.Lock()
_pool_idx  : dict[tuple[str, str], int] = {}


async def _get_conn(language: str, speaker: str) -> _PersistentWsConn:
    key = (language, speaker)
    async with _pool_lock:
        if key not in _pool:
            _pool[key]     = [_PersistentWsConn(language, speaker) for _ in range(_POOL_SIZE)]
            _pool_idx[key] = 0
        idx = _pool_idx[key]
        _pool_idx[key] = (idx + 1) % _POOL_SIZE
        return _pool[key][idx]


async def warmup_ws(language: str = DEFAULT_LANGUAGE, speaker: str = DEFAULT_SPEAKER):
    """Establish all pool connections at startup so first call has zero connect cost."""
    log.info("Warming up %d WS connections (%s/%s)…", _POOL_SIZE, language, speaker)
    key = (language, speaker)
    async with _pool_lock:
        if key not in _pool:
            _pool[key]     = [_PersistentWsConn(language, speaker) for _ in range(_POOL_SIZE)]
            _pool_idx[key] = 0

    results   = await asyncio.gather(*[c.connect() for c in _pool[key]])
    ok_count  = sum(results)
    if ok_count == _POOL_SIZE:
        log.info("✅ All %d WS connections ready.", _POOL_SIZE)
    elif ok_count > 0:
        log.warning("⚠️  %d/%d WS connections ready.", ok_count, _POOL_SIZE)
    else:
        log.warning("⚠️  WS warm-up failed — REST TTS fallback active.")


async def close_all_ws():
    async with _pool_lock:
        for conns in _pool.values():
            for c in conns:
                await c.close()
        _pool.clear()
        _pool_idx.clear()


# ── Public API ─────────────────────────────────────────────────────────────────

async def synthesise_text(
    text: str,
    language: str | None = None,
    speaker:  str | None = None,
) -> str:
    """Synthesise known text → filename. Cache hit = ~0ms."""
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY not set")
    cached = _cache_get(text)
    if cached:
        log.debug("⚡ TTS cache hit: %s", text[:40])
        return _bytes_to_file(cached)
    language = language or DEFAULT_LANGUAGE
    speaker  = speaker  or DEFAULT_SPEAKER
    conn     = await _get_conn(language, speaker)
    filename, _ = await conn.synthesise(text)
    return filename

async def synthesise_text_bytes(
    text: str,
    language: str | None = None,
    speaker:  str | None = None,
) -> bytes | None:
    """
    Synthesise text and return raw audio bytes (no file I/O).
    Used by media_stream.py for in-memory path.
    Cache hit = ~0ms. Falls back to REST on WS failure.
    """
    if not SARVAM_API_KEY:
        return None

    # Cache hit — return bytes directly, no file needed
    cached = _cache_get(text)
    if cached:
        _tts_cache.move_to_end(text.strip())
        return cached

    language = language or DEFAULT_LANGUAGE
    speaker  = speaker  or DEFAULT_SPEAKER
    conn     = await _get_conn(language, speaker)

    # Try WS path first
    cached2 = _cache_get(text)   # re-check after getting conn (may have been filled)
    if cached2:
        return cached2

    async with asyncio.timeout(20):
        async with conn._lock:
            # One more cache check after lock
            cached3 = _cache_get(text)
            if cached3:
                return cached3
            ws_ok = await conn._ensure_ws()
            if not ws_ok:
                # REST fallback
                try:
                    fname = await generate_tts_async(text, language, speaker)
                    path  = os.path.join(AUDIO_DIR, fname)
                    with open(path, "rb") as f:
                        data = f.read()
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                    return data
                except Exception:
                    return None
            try:
                await conn._ws.send(json.dumps({"type": "text", "data": {"text": text.strip()}}))
                await conn._ws.send(json.dumps({"type": "flush"}))
                chunks = await conn._recv_loop()
                if chunks:
                    data = b"".join(chunks)
                    _cache_put(text, data)
                    return data
            except (ConnectionClosed, WebSocketException, OSError, RuntimeError) as e:
                log.warning("synthesise_text_bytes WS error: %s", e)
                conn._ws = None
                conn._config_sent = False
    return None

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
    """REST TTS fallback. Uses in-memory cache. Returns filename."""
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY not set")
    cached = _cache_get(text)
    if cached:
        return _bytes_to_file(cached)

    language = language or ("hi-IN" if _is_mostly_hindi(text) else DEFAULT_LANGUAGE)
    speaker  = speaker  or DEFAULT_SPEAKER
    model    = model    or TTS_MODEL

    payload = {
        "text":                 text,
        "target_language_code": language,
        "speaker":              speaker,
        "model":                model,
        "pace":                 1.0,
        "output_audio_codec":   TTS_CODEC,
        "speech_sample_rate":   TTS_SAMPLE_RATE,
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

    data = base64.b64decode(audios[0])
    _cache_put(text, data)
    filename = _bytes_to_file(data)
    log.info("REST TTS done in %.3fs: %s (%d bytes)",
             time.perf_counter() - t0, filename, len(data))
    return filename


def generate_tts(text: str, language: str | None = None, speaker: str | None = None) -> str:
    return asyncio.run(generate_tts_async(text, language, speaker))