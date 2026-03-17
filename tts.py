"""
tts.py — Persistent WebSocket TTS with keepalive pings.

NEW IN THIS VERSION:
  synthesise_stream_chunks() — async generator that yields MP3 audio chunks
  as they arrive from the Sarvam WebSocket while the LLM is still generating.
  This enables true parallel LLM+TTS streaming: the first audio chunk reaches
  Twilio before the LLM has even finished its response.

  Pipeline with streaming:
    LLM token 1 → queue → WS send → Sarvam generates → audio chunk 1 yielded
    LLM token 2 → queue → ...      (in parallel)
    ...
    First audio playing on Twilio while LLM still generating last sentence.

FIXES RETAINED FROM PREVIOUS VERSION:
  1. synthesise_to_bytes: empty/whitespace key collision fixed.
  2. _do_synthesise_stream: asyncio.Queue replaces spin-wait.
  3. Keepalive ping uses _ping_lock, not _lock (no mid-synthesis block).
  4. synthesise_to_bytes uses REST cache path directly.
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

TTS_MODEL   = "bulbul:v3"
TTS_SPEAKER = "simran"
TTS_WS_URL  = f"wss://api.sarvam.ai/text-to-speech/ws?model={TTS_MODEL}&send_completion_event=true"

AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_files")
os.makedirs(AUDIO_DIR, exist_ok=True)

DEFAULT_SPEAKER  = TTS_SPEAKER
DEFAULT_LANGUAGE = "hi-IN"

_SENTENCE_END   = re.compile(r"[।.?!\n]")
_SOFT_FLUSH_LEN = 50
_PING_INTERVAL  = 25

_TTS_RESPONSE_CACHE_MAX = 50
_tts_response_cache: OrderedDict[str, bytes] = OrderedDict()


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
    return getattr(ws, "open", False)


class _PersistentWsConn:
    def __init__(self, language: str, speaker: str):
        self.language     = language
        self.speaker      = speaker
        self._ws          = None
        self._lock        = asyncio.Lock()
        self._ping_lock   = asyncio.Lock()
        self._keepalive_task: asyncio.Task | None = None
        self._config_sent = False

    async def connect(self) -> bool:
        if _ws_is_open(self._ws):
            return True
        return await self._connect()

    async def synthesise(self, text: str) -> tuple[str, str]:
        async with asyncio.timeout(20):
            async with self._lock:
                return await self._do_synthesise_text(text)

    async def synthesise_stream(self, groq_stream) -> tuple[str, str]:
        async with asyncio.timeout(25):
            async with self._lock:
                return await self._do_synthesise_stream(groq_stream)

    async def synthesise_stream_chunks(self, groq_stream):
        """
        NEW: Async generator that yields raw MP3 bytes chunks as they arrive
        from the Sarvam WebSocket, while the LLM is still generating tokens.

        Usage:
            async for mp3_chunk in conn.synthesise_stream_chunks(groq_stream):
                await stream_to_twilio(mp3_chunk)

        This is the true low-latency path:
          - LLM tokens → asyncio.Queue → WS send → Sarvam audio → yield chunk
          - Caller receives and streams first audio chunk in ~0.8-1s
          - Subsequent chunks arrive while first chunk is already playing
        """
        async with asyncio.timeout(30):
            async with self._lock:
                async for chunk in self._do_synthesise_stream_chunks(groq_stream):
                    yield chunk

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
            "data": {"target_language_code": self.language, "speaker": self.speaker, "pace": 1.0}
        }))
        self._config_sent = True
        log.info("WS config sent (lang=%s speaker=%s)", self.language, self.speaker)

    async def _keepalive_loop(self):
        try:
            while True:
                await asyncio.sleep(_PING_INTERVAL)
                if not _ws_is_open(self._ws):
                    log.info("Keepalive: WS closed — stopping")
                    return
                try:
                    async with self._ping_lock:
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
        log.info("WS not ready — reconnecting...")
        return await self._connect()

    async def _do_synthesise_text(self, text: str) -> tuple[str, str]:
        t0    = time.perf_counter()
        ws_ok = await self._ensure_ws()
        if not ws_ok:
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
                    log.info("⏱  TTS done (text): %.3fs", time.perf_counter() - t0)
                    break
                elif msg_type == "error":
                    raise RuntimeError(f"Sarvam WS error: {data}")
            if audio_chunks:
                return _save_audio(audio_chunks, text, t0)
            log.warning("WS: no audio — REST fallback")
        except (ConnectionClosed, WebSocketException, OSError, RuntimeError) as e:
            log.warning("WS error (text synth): %s — REST fallback", e)
            self._ws = None
            self._config_sent = False
        fname = await generate_tts_async(text, self.language, self.speaker)
        return fname, text

    async def _do_synthesise_stream(self, groq_stream) -> tuple[str, str]:
        """Collect all stream audio then return — used by llm_to_tts_stream()."""
        t_total      = time.perf_counter()
        audio_chunks : list[bytes] = []
        full_text    = ""
        token_queue  : asyncio.Queue[str | None] = asyncio.Queue()

        async def _drain_llm():
            nonlocal full_text
            first = True
            async for chunk in groq_stream:
                token = chunk.choices[0].delta.content or ""
                if not token:
                    continue
                full_text += token
                await token_queue.put(token)
                if first:
                    log.info("⏱  LLM first token: %.3fs", time.perf_counter() - t_total)
                    first = False
            log.info("⏱  LLM done: %.3fs | %s", time.perf_counter() - t_total, full_text[:60])
            await token_queue.put(None)

        llm_task = asyncio.create_task(_drain_llm())
        ws_ok    = await self._ensure_ws()

        if ws_ok:
            try:
                async def _send_from_queue():
                    buf = ""
                    while True:
                        token = await token_queue.get()
                        if token is None:
                            break
                        buf += token
                        if _SENTENCE_END.search(buf) or len(buf) >= _SOFT_FLUSH_LEN:
                            await self._ws.send(json.dumps({"type": "text", "data": {"text": buf.strip()}}))
                            buf = ""
                    if buf.strip():
                        await self._ws.send(json.dumps({"type": "text", "data": {"text": buf.strip()}}))
                    await self._ws.send(json.dumps({"type": "flush"}))
                    log.info("⏱  WS flush: %.3fs", time.perf_counter() - t_total)

                async def _recv_audio():
                    async for msg in self._ws:
                        data     = json.loads(msg)
                        msg_type = data.get("type")
                        if msg_type == "audio":
                            b64 = (data.get("data") or {}).get("audio") or ""
                            if b64:
                                audio_chunks.append(base64.b64decode(b64))
                        elif msg_type == "event":
                            log.info("⏱  TTS complete: %.3fs", time.perf_counter() - t_total)
                            break
                        elif msg_type == "error":
                            raise RuntimeError(f"Sarvam WS error: {data}")

                await asyncio.gather(llm_task, _send_from_queue(), _recv_audio())
                full_text = full_text.strip() or "Kripya dobara bolein."
                if audio_chunks:
                    return _save_audio(audio_chunks, full_text, t_total)
                log.warning("WS: no audio — REST fallback")
            except (ConnectionClosed, WebSocketException, OSError, RuntimeError) as e:
                log.warning("WS stream error: %s — REST fallback", e)
                self._ws = None
                self._config_sent = False

        if not llm_task.done():
            await llm_task
        full_text = full_text.strip() or "Kripya dobara bolein."
        fname = await generate_tts_async(full_text, self.language, self.speaker)
        return fname, full_text

    async def _do_synthesise_stream_chunks(self, groq_stream):
        """
        Per-sentence streaming generator.

        For each sentence from the LLM:
          1. Send text + flush to Sarvam WS
          2. Collect all audio events until the completion event
          3. Join those bytes → one valid MP3 for that sentence
             (Sarvam audio events within a single flush are sequential MP3
              frames of one continuous synthesis — joining them is correct)
          4. Yield that MP3 immediately to the caller

        This gives both low latency (first sentence ~0.8s) and clean audio
        (no boundary crackling — each yielded chunk is a complete MP3).

        Timeline:
          t=0.0  LLM first token
          t=0.3  First sentence boundary → flush to Sarvam
          t=0.8  Sarvam completion event → yield sentence-1 MP3 → Twilio plays
          t=0.4  Second sentence ready → flush → audio → yield → Twilio
        """
        t_total   = time.perf_counter()
        full_text = ""
        token_queue: asyncio.Queue[str | None] = asyncio.Queue()

        # Task A: drain LLM tokens into queue in background
        async def _drain_llm():
            nonlocal full_text
            first = True
            try:
                async for chunk in groq_stream:
                    token = (chunk.choices[0].delta.content or "") if chunk.choices else ""
                    if not token:
                        continue
                    full_text += token
                    await token_queue.put(token)
                    if first:
                        log.info("⏱  LLM first token: %.3fs", time.perf_counter() - t_total)
                        first = False
            finally:
                await token_queue.put(None)
            log.info("⏱  LLM done: %.3fs | %r", time.perf_counter() - t_total, full_text[:80])

        ws_ok = await self._ensure_ws()

        if not ws_ok:
            # No WS — drain LLM then REST fallback
            llm_task = asyncio.create_task(_drain_llm())
            await llm_task
            full_text = full_text.strip() or "Sorry, main samajh nahi payi."
            try:
                fname = await generate_tts_async(full_text, self.language, self.speaker)
                filepath = os.path.join(AUDIO_DIR, fname)
                with open(filepath, "rb") as f:
                    data = f.read()
                yield data
            except Exception as e:
                log.warning("REST TTS fallback failed: %s", e)
            return

        llm_task    = asyncio.create_task(_drain_llm())
        buf         = ""
        first_audio = True
        ws_error    = False

        try:
            while True:
                token         = await token_queue.get()
                end_of_stream = (token is None)

                if not end_of_stream:
                    buf += token
                    # Keep buffering until sentence boundary or soft-flush threshold
                    if not (_SENTENCE_END.search(buf) or len(buf) >= _SOFT_FLUSH_LEN):
                        continue

                flush_text = buf.strip()
                buf = ""

                if flush_text:
                    # Send sentence + flush, then collect all audio for this flush
                    await self._ws.send(json.dumps({
                        "type": "text",
                        "data": {"text": flush_text},
                    }))
                    await self._ws.send(json.dumps({"type": "flush"}))
                    log.info("⏱  WS sentence flushed (%d chars): %.3fs",
                             len(flush_text), time.perf_counter() - t_total)

                    audio_chunks: list[bytes] = []
                    async for msg in self._ws:
                        data     = json.loads(msg)
                        msg_type = data.get("type")
                        if msg_type == "audio":
                            b64 = (data.get("data") or {}).get("audio") or ""
                            if b64:
                                if not audio_chunks and first_audio:
                                    log.info("⏱  First audio chunk: %.3fs",
                                             time.perf_counter() - t_total)
                                    first_audio = False
                                audio_chunks.append(base64.b64decode(b64))
                        elif msg_type == "event":
                            log.info("⏱  Sentence audio complete: %.3fs",
                                     time.perf_counter() - t_total)
                            break
                        elif msg_type == "error":
                            raise RuntimeError(f"Sarvam WS error: {data}")

                    if audio_chunks:
                        # All audio events for one flush = sequential MP3 frames
                        # of a single synthesis → joining them gives one valid MP3
                        yield b"".join(audio_chunks)

                if end_of_stream:
                    break

            log.info("⏱  TTS stream complete: %.3fs", time.perf_counter() - t_total)

        except (ConnectionClosed, WebSocketException, OSError, RuntimeError) as e:
            log.warning("WS stream chunk error: %s — REST fallback", e)
            self._ws = None
            self._config_sent = False
            ws_error = True
            while not token_queue.empty():
                token_queue.get_nowait()

        await asyncio.gather(llm_task, return_exceptions=True)

        # REST fallback only if WS failed before any audio was yielded
        if ws_error and first_audio:
            log.info("Streaming REST fallback for: %r", full_text[:60])
            full_text = full_text.strip() or "Sorry, main samajh nahi payi."
            try:
                fname    = await generate_tts_async(full_text, self.language, self.speaker)
                filepath = os.path.join(AUDIO_DIR, fname)
                with open(filepath, "rb") as f:
                    data = f.read()
                yield data
            except Exception as e:
                log.warning("REST TTS fallback also failed: %s", e)

    async def synthesise_to_bytes(self, text: str) -> bytes | None:
        cleaned = text.strip()
        if not cleaned:
            log.debug("synthesise_to_bytes: empty text — skipping")
            return None

        if cleaned in _tts_response_cache:
            _tts_response_cache.move_to_end(cleaned)
            return _tts_response_cache[cleaned]

        try:
            fname = await generate_tts_async(cleaned, self.language, self.speaker)
            path  = os.path.join(AUDIO_DIR, fname)
            with open(path, "rb") as f:
                data = f.read()
            try:
                os.remove(path)
            except OSError:
                pass
            _tts_response_cache[cleaned] = data
            _tts_response_cache.move_to_end(cleaned)
            while len(_tts_response_cache) > _TTS_RESPONSE_CACHE_MAX:
                _tts_response_cache.popitem(last=False)
            return data
        except Exception as e:
            log.warning("synthesise_to_bytes REST failed: %s", e)
            return None


def _save_audio(chunks: list[bytes], full_text: str, t0: float) -> tuple[str, str]:
    combined = b"".join(chunks)
    filename = f"{uuid.uuid4()}.mp3"
    with open(os.path.join(AUDIO_DIR, filename), "wb") as f:
        f.write(combined)
    log.info("⏱  TOTAL (WS): %.3fs | %d bytes", time.perf_counter() - t0, len(combined))
    return filename, full_text


# ── Connection pool ────────────────────────────────────────────────────────────

_pool:      dict[tuple[str, str], _PersistentWsConn] = {}
_pool_lock = asyncio.Lock()


async def _get_conn(language: str, speaker: str) -> _PersistentWsConn:
    key = (language, speaker)
    async with _pool_lock:
        if key not in _pool:
            _pool[key] = _PersistentWsConn(language, speaker)
        return _pool[key]


async def warmup_ws(language: str = DEFAULT_LANGUAGE, speaker: str = DEFAULT_SPEAKER):
    log.info("Warming up persistent WS TTS (%s / %s)...", language, speaker)
    conn = await _get_conn(language, speaker)
    ok   = await conn.connect()
    if ok:
        log.info("✅ WS TTS ready — keepalive active.")
    else:
        log.warning("⚠️  WS warm-up failed — REST fallback active.")


async def close_all_ws():
    async with _pool_lock:
        for conn in _pool.values():
            await conn.close()
        _pool.clear()


# ── Public API ─────────────────────────────────────────────────────────────────

async def llm_to_tts_stream(groq_stream, language=None, speaker=None) -> tuple[str, str]:
    """Collect full stream then return filename. Non-streaming path."""
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY not set")
    conn = await _get_conn(language or DEFAULT_LANGUAGE, speaker or DEFAULT_SPEAKER)
    return await conn.synthesise_stream(groq_stream)


async def llm_to_tts_stream_chunks(groq_stream, language=None, speaker=None):
    """
    NEW PUBLIC API — async generator yielding MP3 chunks as they arrive.
    Use this for the lowest possible latency path in media_stream.py.

    Example:
        async for mp3_chunk in llm_to_tts_stream_chunks(groq_stream):
            mulaw = convert_to_mulaw(mp3_chunk)
            await send_to_twilio(mulaw)
    """
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY not set")
    conn = await _get_conn(language or DEFAULT_LANGUAGE, speaker or DEFAULT_SPEAKER)
    async for chunk in conn.synthesise_stream_chunks(groq_stream):
        yield chunk


async def synthesise_text(text: str, language=None, speaker=None) -> str:
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY not set")

    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("synthesise_text called with empty string — check caller")

    if cleaned in _tts_response_cache:
        audio_bytes = _tts_response_cache[cleaned]
        _tts_response_cache.move_to_end(cleaned)
        filename = f"{uuid.uuid4()}.mp3"
        with open(os.path.join(AUDIO_DIR, filename), "wb") as f:
            f.write(audio_bytes)
        log.debug("TTS cache hit: %s", cleaned[:40])
        return filename

    conn = await _get_conn(language or DEFAULT_LANGUAGE, speaker or DEFAULT_SPEAKER)
    filename, _ = await conn.synthesise(cleaned)
    try:
        with open(os.path.join(AUDIO_DIR, filename), "rb") as f:
            audio_bytes = f.read()
        _tts_response_cache[cleaned] = audio_bytes
        _tts_response_cache.move_to_end(cleaned)
        while len(_tts_response_cache) > _TTS_RESPONSE_CACHE_MAX:
            _tts_response_cache.popitem(last=False)
    except OSError:
        pass
    return filename


async def synthesise_text_bytes(text: str, language=None, speaker=None) -> bytes | None:
    if not SARVAM_API_KEY:
        return None
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    if cleaned in _tts_response_cache:
        _tts_response_cache.move_to_end(cleaned)
        return _tts_response_cache[cleaned]
    conn = await _get_conn(language or DEFAULT_LANGUAGE, speaker or DEFAULT_SPEAKER)
    return await conn.synthesise_to_bytes(cleaned)


async def close_http_client():
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
    await close_all_ws()


# ── REST fallback ──────────────────────────────────────────────────────────────

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
    return sum(1 for c in letters if "\u0900" <= c <= "\u097F") / len(letters) > 0.25


async def generate_tts_async(text: str, language=None, speaker=None, model=None) -> str:
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY not set")

    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("generate_tts_async called with empty string")

    if cleaned in _tts_response_cache:
        audio_bytes = _tts_response_cache[cleaned]
        _tts_response_cache.move_to_end(cleaned)
        filename = f"{uuid.uuid4()}.mp3"
        with open(os.path.join(AUDIO_DIR, filename), "wb") as f:
            f.write(audio_bytes)
        return filename

    language = language or ("hi-IN" if _is_mostly_hindi(cleaned) else DEFAULT_LANGUAGE)
    speaker  = speaker  or DEFAULT_SPEAKER
    model    = model    or TTS_MODEL

    t0 = time.perf_counter()

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = await _get_http_client().post(
                TTS_REST_URL,
                json={
                    "text": cleaned,
                    "target_language_code": language,
                    "speaker": speaker,
                    "model": model,
                    "pace": 1.0,
                    "output_audio_codec": "mp3",
                },
                headers={
                    "api-subscription-key": SARVAM_API_KEY,
                    "Content-Type": "application/json",
                },
            )
            if resp.status_code == 403:
                raise ValueError("Sarvam 403 — check SARVAM_API_KEY")
            resp.raise_for_status()
            break
        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500 or attempt == 2:
                raise
            last_exc = e
            wait = 0.5 * (2 ** attempt)
            log.warning("Sarvam TTS %d error (attempt %d/3) — retrying in %.1fs",
                        e.response.status_code, attempt + 1, wait)
            await asyncio.sleep(wait)
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            if attempt == 2:
                raise
            last_exc = e
            wait = 0.5 * (2 ** attempt)
            log.warning("Sarvam TTS network error (attempt %d/3) — retrying in %.1fs",
                        attempt + 1, wait)
            await asyncio.sleep(wait)

    audios = resp.json().get("audios") or []
    if not audios:
        raise ValueError("Sarvam REST TTS returned no audio")

    audio_bytes = base64.b64decode(audios[0])
    _tts_response_cache[cleaned] = audio_bytes
    _tts_response_cache.move_to_end(cleaned)
    while len(_tts_response_cache) > _TTS_RESPONSE_CACHE_MAX:
        _tts_response_cache.popitem(last=False)

    filename = f"{uuid.uuid4()}.mp3"
    with open(os.path.join(AUDIO_DIR, filename), "wb") as f:
        f.write(audio_bytes)
    log.info("REST TTS: %.3fs | %d bytes", time.perf_counter() - t0, len(audio_bytes))
    return filename


def generate_tts(text: str, language=None, speaker=None) -> str:
    """Sync wrapper — FOR CLI/SCRIPTS ONLY."""
    return asyncio.run(generate_tts_async(text, language, speaker))