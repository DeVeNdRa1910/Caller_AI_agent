"""
tts.py — Streaming LLM→TTS pipeline using Sarvam WebSocket TTS.

Key design:
- `llm_to_tts_stream(groq_stream, language)` consumes a Groq async token
  stream, pipes text into a Sarvam WebSocket TTS connection in real time,
  and yields raw MP3 bytes as they arrive.
- `generate_tts_async(text)` is the simple non-streaming fallback (same REST
  API as before, used for outbound opening messages).
- Sarvam WebSocket: wss://api.sarvam.ai/text-to-speech/ws?model=bulbul:v3
  Protocol: config → text chunks → flush → receive audio chunks until done.
"""

import asyncio
import base64
import json
import logging
import os
import re
import uuid
import httpx
import websockets

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

SARVAM_API_KEY = (os.getenv("SARVAM_API_KEY") or "").strip()
TTS_REST_URL   = "https://api.sarvam.ai/text-to-speech"
TTS_WS_URL     = "wss://api.sarvam.ai/text-to-speech/ws?model=bulbul:v3"

AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_files")
os.makedirs(AUDIO_DIR, exist_ok=True)

DEFAULT_SPEAKER  = "simran"
DEFAULT_LANGUAGE = "hi-IN"

# ── Persistent REST client (used only for generate_tts_async fallback) ────────
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


def _is_mostly_hindi(text: str) -> bool:
    letters = [c for c in text if c.strip()]
    if not letters:
        return False
    dev = sum(1 for c in letters if "\u0900" <= c <= "\u097F")
    return (dev / len(letters)) > 0.25


# ── WebSocket streaming pipeline ──────────────────────────────────────────────

async def llm_to_tts_stream(
    groq_stream,                    # AsyncGroq streaming completion
    language: str | None = None,
    speaker:  str | None = None,
) -> str:
    """
    Pipeline:
      Groq token stream → Sarvam WebSocket TTS → MP3 file.

    Text is sent to Sarvam as tokens arrive from Groq (character-by-character
    buffered at ~30 chars so Sarvam can start generating audio while Groq is
    still producing text).

    Returns: saved MP3 filename under AUDIO_DIR.
    Latency: TTS starts ~30 chars into LLM generation (~0.3-0.5s into stream).
    """
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY not set")

    language = language or DEFAULT_LANGUAGE
    speaker  = speaker  or DEFAULT_SPEAKER

    audio_chunks: list[bytes] = []
    full_text = ""

    ws_headers = {"Api-Subscription-Key": SARVAM_API_KEY}

    try:
        async with websockets.connect(
            TTS_WS_URL,
            additional_headers=ws_headers,
            open_timeout=8,
            close_timeout=5,
        ) as ws:

            # 1. Send config (must be first message)
            await ws.send(json.dumps({
                "type": "config",
                "data": {
                    "target_language_code": language,
                    "speaker": speaker,
                    "model": "bulbul:v3",
                    "output_audio_codec": "mp3",
                    "pace": 1.0,
                    "min_buffer_size": 30,   # start audio after 30 chars buffered
                    "send_completion_event": True,
                }
            }))

            # 2. Consume Groq stream and pipe text to Sarvam in real time
            async def _send_llm_tokens():
                nonlocal full_text
                buf = ""
                async for chunk in groq_stream:
                    token = chunk.choices[0].delta.content or ""
                    full_text += token
                    buf += token
                    # Send to Sarvam in ~50-char chunks to avoid too many small messages
                    if len(buf) >= 50:
                        await ws.send(json.dumps({
                            "type": "text",
                            "data": {"text": buf}
                        }))
                        buf = ""
                # Send any remaining text
                if buf:
                    await ws.send(json.dumps({
                        "type": "text",
                        "data": {"text": buf}
                    }))
                # Flush — tells Sarvam to process remaining buffer and finish
                await ws.send(json.dumps({"type": "flush"}))

            # 3. Receive audio chunks concurrently while LLM is streaming
            async def _receive_audio():
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                    except Exception:
                        continue
                    msg_type = data.get("type")
                    if msg_type == "audio":
                        b64 = (data.get("data") or {}).get("audio") or ""
                        if b64:
                            audio_chunks.append(base64.b64decode(b64))
                    elif msg_type == "event":
                        # completion event — TTS is done
                        break
                    elif msg_type == "error":
                        raise RuntimeError(f"Sarvam WS error: {data}")

            # Run both concurrently: send tokens while receiving audio
            await asyncio.gather(_send_llm_tokens(), _receive_audio())

    except Exception as e:
        log.warning("WebSocket TTS failed (%s), falling back to REST", e)
        # Fallback: if we got full_text from LLM but WS failed, use REST
        if full_text.strip():
            return await generate_tts_async(full_text.strip(), language=language, speaker=speaker)
        raise

    full_text = full_text.strip() or "कृपया दोबारा बोलें।"
    log.info("WS TTS complete: %d audio chunks, text='%s'", len(audio_chunks), full_text[:80])

    if not audio_chunks:
        log.warning("No audio from WebSocket, falling back to REST for: %s", full_text[:60])
        return await generate_tts_async(full_text, language=language, speaker=speaker)

    combined = b"".join(audio_chunks)
    filename = f"{uuid.uuid4()}.mp3"
    with open(os.path.join(AUDIO_DIR, filename), "wb") as f:
        f.write(combined)

    return filename, full_text   # return tuple so caller can save history


# ── REST fallback (used for outbound opening & if WS fails) ───────────────────

async def generate_tts_async(
    text: str,
    language: str | None = None,
    speaker:  str | None = None,
) -> str:
    """REST-based TTS. Returns filename only (no text). Used for static messages."""
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY not set")

    language = language or ("hi-IN" if _is_mostly_hindi(text) else DEFAULT_LANGUAGE)
    speaker  = speaker  or DEFAULT_SPEAKER

    payload = {
        "text": text,
        "target_language_code": language,
        "speaker": speaker,
        "model": "bulbul:v3",
        "output_audio_codec": "mp3",
        "pace": 1.0,
    }
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }

    client = _get_http_client()
    resp   = await client.post(TTS_REST_URL, json=payload, headers=headers)

    if not resp.is_success:
        try:
            err = resp.json()
        except Exception:
            err = resp.text or f"HTTP {resp.status_code}"
        if resp.status_code == 403:
            raise ValueError(f"Sarvam 403: invalid/expired key. {err}")
        resp.raise_for_status()

    data   = resp.json()
    audios = data.get("audios") or []
    if not audios:
        raise ValueError("Sarvam REST TTS returned no audio")

    audio_bytes = base64.b64decode(audios[0])
    filename    = f"{uuid.uuid4()}.mp3"
    with open(os.path.join(AUDIO_DIR, filename), "wb") as f:
        f.write(audio_bytes)

    log.info("REST TTS done: %s (%d bytes)", filename, len(audio_bytes))
    return filename


def generate_tts(text: str, language: str | None = None, speaker: str | None = None) -> str:
    """Sync wrapper — only for scripts outside async context."""
    import asyncio
    return asyncio.run(generate_tts_async(text, language, speaker))