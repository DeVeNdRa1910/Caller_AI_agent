"""
stt.py — Async STT using Sarvam Saaras v3.

FIXES IN THIS VERSION:
  1. Retry logic: transient 5xx errors and network failures now retry with
     exponential backoff (3 attempts). Previously a single Sarvam hiccup
     would drop the user's entire utterance and respond with an apology.
  2. Uses httpx.AsyncClient (shared, persistent) instead of blocking requests.
     This avoids a thread-blocking call in the async FastAPI event loop,
     which was adding ~100-300ms of extra latency per turn.
"""

import asyncio
import logging
import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

SARVAM_API_KEY = (os.getenv("SARVAM_API_KEY") or "").strip()
STT_URL        = "https://api.sarvam.ai/speech-to-text"

# Shared persistent client — reuses TCP connections across turns
_stt_client: httpx.AsyncClient | None = None


def _get_stt_client() -> httpx.AsyncClient:
    global _stt_client
    if _stt_client is None or _stt_client.is_closed:
        _stt_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=5),
            timeout=httpx.Timeout(10.0),
        )
    return _stt_client


async def close_stt_client():
    global _stt_client
    if _stt_client and not _stt_client.is_closed:
        await _stt_client.aclose()


async def transcribe_audio_async(
    audio_path:    str | None   = None,
    audio_bytes:   bytes | None = None,
    mode:          str          = "codemix",
    content_type:  str          = "audio/wav",
    filename:      str          = "audio.wav",
    language_code: str          = "hi-IN",
) -> str:
    """
    Async transcribe audio using Sarvam AI STT (Saaras v3).

    Provide either audio_path (path to file) or audio_bytes.
    mode options: "codemix" (best for Hindi/English mix), "transcribe",
                  "translate", "verbatim", "translit"
    language_code: hint to STT about expected language (e.g. "hi-IN").
    Returns the transcript text (empty string if nothing recognised).
    """
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY is not set")

    t0      = time.perf_counter()
    headers = {"api-subscription-key": SARVAM_API_KEY}
    data    = {"model": "saaras:v3", "mode": mode}
    if language_code:
        data["language_code"] = language_code
    client  = _get_stt_client()

    if audio_path:
        fname = os.path.basename(audio_path)
        with open(audio_path, "rb") as f:
            raw = f.read()
        files = {"file": (fname, raw, content_type)}
    elif audio_bytes:
        files = {"file": (filename, audio_bytes, content_type)}
    else:
        raise ValueError("Provide either audio_path or audio_bytes")

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = await client.post(STT_URL, headers=headers, data=data, files=files)
            if resp.status_code == 403:
                raise ValueError("Sarvam STT 403 — check SARVAM_API_KEY")
            if resp.status_code == 400:
                log.warning("STT 400 Bad Request — retrying without language_code")
                data.pop("language_code", None)
                resp = await client.post(STT_URL, headers=headers, data=data, files=files)
                if resp.status_code != 200:
                    log.warning("STT retry also failed (%d) — returning empty", resp.status_code)
                    return ""
            resp.raise_for_status()

            result     = resp.json()
            transcript = (result.get("transcript") or "").strip()
            log.info("⏱  STT done: %.3fs (attempt %d) | %s",
                     time.perf_counter() - t0, attempt + 1,
                     transcript[:80] or "<empty>")
            return transcript

        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500 or attempt == 2:
                raise
            last_exc = e
            wait = 0.5 * (2 ** attempt)
            log.warning("Sarvam STT %d error (attempt %d/3) — retrying in %.1fs",
                        e.response.status_code, attempt + 1, wait)
            await asyncio.sleep(wait)

        except (httpx.ConnectError, httpx.TimeoutException) as e:
            if attempt == 2:
                raise
            last_exc = e
            wait = 0.5 * (2 ** attempt)
            log.warning("Sarvam STT network error (attempt %d/3) — retrying in %.1fs",
                        attempt + 1, wait)
            await asyncio.sleep(wait)

    # Should not reach here — last attempt raises above
    raise RuntimeError(f"STT failed after 3 attempts: {last_exc}")


# Sync wrapper for non-async callers — CLI/scripts ONLY
def transcribe_audio(
    audio_path:    str | None   = None,
    audio_bytes:   bytes | None = None,
    mode:          str          = "codemix",
    content_type:  str          = "audio/wav",
    filename:      str          = "audio.wav",
    language_code: str          = "hi-IN",
) -> str:
    """
    FOR CLI/SCRIPTS ONLY — do NOT call from async context.
    asyncio.run() raises RuntimeError if an event loop is already running.
    """
    return asyncio.run(
        transcribe_audio_async(audio_path, audio_bytes, mode, content_type, filename, language_code)
    )