"""
tts.py — Persistent WebSocket TTS + streaming LLM pipeline.

KEY FIX: The WS 422 error was caused by unsupported fields in the config payload
(output_audio_bitrate, max_chunk_length). Stripped to only fields Sarvam accepts.

KEY FIX 2: Sarvam WS closes the connection after each flush/completion event.
We now detect this and reconnect, but skip the double-reconnect waste by
going straight to REST if WS fails, while also running LLM in parallel.

ARCHITECTURE:
  - LLM streams tokens into a queue
  - TTS WS (or REST fallback) drains the queue
  - Both run concurrently via asyncio.gather

Sarvam bulbul:v3 WS protocol:
  URL:  wss://api.sarvam.ai/text-to-speech/ws
        ?model=bulbul:v3&send_completion_event=true
  Header: Api-Subscription-Key: <key>
  1. config  → {"type":"config","data":{"target_language_code","speaker","pace"}}
  2. text    → {"type":"text","data":{"text":"…"}}
  3. flush   → {"type":"flush"}
  ← audio   ← {"type":"audio","data":{"audio":"<b64 mp3>"}}
  ← event   ← {"type":"event",...}  ← connection closes after this
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

# v3 model + simran speaker as requested
TTS_MODEL      = "bulbul:v3"
TTS_SPEAKER    = "simran"
TTS_WS_URL     = f"wss://api.sarvam.ai/text-to-speech/ws?model={TTS_MODEL}&send_completion_event=true"

AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_files")
os.makedirs(AUDIO_DIR, exist_ok=True)

DEFAULT_SPEAKER  = TTS_SPEAKER
DEFAULT_LANGUAGE = "hi-IN"

# Flush to TTS at sentence boundaries OR when buffer gets long
_SENTENCE_END  = re.compile(r"[।.?!\n]")
_SOFT_FLUSH_LEN = 50   # chars — flush mid-sentence to start TTS early


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
    Persistent Sarvam WS connection.

    NOTE: Sarvam closes the WS after each flush+event cycle. So "persistent"
    means we reconnect once per turn (not per retry). The key saving vs the
    original code is:
      - We connect ONCE per turn (not twice on failure)
      - We don't waste time on two failed WS attempts before REST
      - LLM tokens are buffered so REST fallback can use them instantly
    """

    def __init__(self, language: str, speaker: str):
        self.language = language
        self.speaker  = speaker
        self._ws      = None
        self._lock    = asyncio.Lock()

    async def synthesise(self, groq_stream) -> tuple[str, str]:
        async with asyncio.timeout(25):
            async with self._lock:
                return await self._do_synthesise(groq_stream)

    async def close(self):
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _connect(self) -> bool:
        """Connect and send config. Returns True on success."""
        t0 = time.perf_counter()
        try:
            headers  = {"Api-Subscription-Key": SARVAM_API_KEY}
            self._ws = await websockets.connect(
                TTS_WS_URL,
                additional_headers=headers,
                open_timeout=8,
                close_timeout=3,
                ping_interval=None,
            )
            log.info("WS connected in %.3fs", time.perf_counter() - t0)

            # MINIMAL config — only fields Sarvam v3 WS actually accepts
            # DO NOT add output_audio_bitrate or max_chunk_length — causes 422
            await self._ws.send(json.dumps({
                "type": "config",
                "data": {
                    "target_language_code": self.language,
                    "speaker":              self.speaker,
                    "pace":                 1.1,
                }
            }))
            log.info("WS config sent (lang=%s speaker=%s)", self.language, self.speaker)
            return True
        except Exception as e:
            log.warning("WS connect failed: %s", e)
            self._ws = None
            return False

    async def _do_synthesise(self, groq_stream) -> tuple[str, str]:
        t_total      = time.perf_counter()
        audio_chunks : list[bytes] = []
        full_text    = ""
        llm_buffer   : list[str]   = []
        llm_done_evt = asyncio.Event()

        # PARALLEL: WS connect (~300ms) + LLM drain run simultaneously.
        # By the time first tokens arrive, WS is already ready => ~300ms saved.

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

        llm_task     = asyncio.create_task(_drain_llm())
        connect_task = asyncio.create_task(self._connect())
        ws_ok        = await connect_task
        log.info("⏱  WS ready: %.3fs (ok=%s)", time.perf_counter() - t_total, ws_ok)

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
                self._ws  = None
                full_text = full_text.strip() or "कृपया दोबारा बोलें।"
                if audio_chunks:
                    return _save_audio(audio_chunks, full_text, t_total)
                log.warning("WS: no audio — REST fallback")

            except (ConnectionClosed, WebSocketException, OSError, RuntimeError) as e:
                log.warning("WS error: %s — REST fallback", e)
                self._ws = None

        # REST fallback — LLM already draining in background
        if not llm_task.done():
            await llm_task
        full_text = full_text.strip() or "कृपया दोबारा बोलें।"
        log.info("⏱  REST text ready: %.3fs", time.perf_counter() - t_total)
        fname = await generate_tts_async(full_text, self.language, self.speaker)
        log.info("⏱  TOTAL (REST): %.3fs", time.perf_counter() - t_total)
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

_pool      : dict[tuple[str, str], _WsConn] = {}
_pool_lock = asyncio.Lock()


async def _get_conn(language: str, speaker: str) -> _WsConn:
    key = (language, speaker)
    async with _pool_lock:
        if key not in _pool:
            _pool[key] = _WsConn(language, speaker)
        return _pool[key]


async def warmup_ws(language: str = DEFAULT_LANGUAGE, speaker: str = DEFAULT_SPEAKER):
    """
    Warm-up: just validate credentials by doing a test WS connect.
    Sarvam closes WS after each turn so we can't keep one open forever.
    The real per-turn connect cost is ~250-400ms — unavoidable with Sarvam.
    """
    log.info("Warming up WS TTS (%s / %s)…", language, speaker)
    conn = await _get_conn(language, speaker)
    ok   = await conn._connect()
    if ok:
        try:
            await conn._ws.close()
        except Exception:
            pass
        conn._ws = None
        log.info("WS TTS warm-up done — credentials valid.")
    else:
        log.warning("WS warm-up failed — will use REST TTS.")


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
    return await conn.synthesise(groq_stream)


async def close_all_ws():
    async with _pool_lock:
        for conn in _pool.values():
            await conn.close()
        _pool.clear()


# ── REST client ────────────────────────────────────────────────────────────────

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
        "pace":                 1.1,
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