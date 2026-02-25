"""
stt.py — Async STT using Sarvam Saaras v3.

Uses httpx.AsyncClient (shared, persistent) instead of blocking requests.
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
            timeout=httpx.Timeout(20.0),
        )
    return _stt_client


async def close_stt_client():
    global _stt_client
    if _stt_client and not _stt_client.is_closed:
        await _stt_client.aclose()


async def transcribe_audio_async(
    audio_path:   str | None   = None,
    audio_bytes:  bytes | None = None,
    mode:         str          = "codemix",
    content_type: str          = "audio/wav",
    filename:     str          = "audio.wav",
) -> str:
    """
    Async transcribe audio using Sarvam AI STT (Saaras v3).

    Provide either audio_path (path to file) or audio_bytes.
    mode options: "codemix" (best for Hindi/English mix), "transcribe",
                  "translate", "verbatim", "translit"
    content_type/filename: use audio/mpeg + "audio.mp3" when passing MP3 bytes.
    Returns the transcript text (empty string if nothing recognised).
    """
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY is not set")

    t0      = time.perf_counter()
    headers = {"api-subscription-key": SARVAM_API_KEY}
    data    = {"model": "saaras:v3", "mode": mode}
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

    resp = await client.post(STT_URL, headers=headers, data=data, files=files)
    resp.raise_for_status()

    result     = resp.json()
    transcript = (result.get("transcript") or "").strip()
    log.info("⏱  STT done: %.3fs | %s", time.perf_counter() - t0, transcript[:80] or "<empty>")
    return transcript


# Sync wrapper for non-async callers
def transcribe_audio(
    audio_path:   str | None   = None,
    audio_bytes:  bytes | None = None,
    mode:         str          = "codemix",
    content_type: str          = "audio/wav",
    filename:     str          = "audio.wav",
) -> str:
    return asyncio.run(
        transcribe_audio_async(audio_path, audio_bytes, mode, content_type, filename)
    )