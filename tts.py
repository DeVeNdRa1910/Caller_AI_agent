import logging
import os
import base64
import uuid
import requests
from dotenv import load_dotenv
load_dotenv()
log = logging.getLogger(__name__)

# Strip whitespace; 403 often caused by trailing space or wrong key
SARVAM_API_KEY = (os.getenv("SARVAM_API_KEY") or "").strip()
TTS_URL = "https://api.sarvam.ai/text-to-speech"

# All TTS files saved here so /audio/{filename} can find them reliably
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_files")

os.makedirs(AUDIO_DIR, exist_ok=True)

# Preferred Indian voice: Priya (Bulbul v3) – clear, natural Indian female. Use en-IN for Indian English accent.
DEFAULT_SPEAKER = "priya"
DEFAULT_LANGUAGE = "en-IN"


def _is_mostly_hindi(text: str) -> bool:
    """True if a significant portion of text is in Devanagari (Hindi)."""
    if not text or not text.strip():
        return False
    letters = [c for c in text if c.strip()]
    if not letters:
        return False
    devanagari_count = sum(1 for c in letters if "\u0900" <= c <= "\u097F")
    return (devanagari_count / len(letters)) > 0.25


def generate_tts(
    text: str,
    language: str | None = None,
    speaker: str | None = None,
) -> str:
    """
    Generate speech from text using Sarvam AI TTS (Bulbul).
    Uses Indian-accent English (en-IN) or Hindi (hi-IN) based on text.
    Returns only the filename (e.g. uuid.mp3) for use in /audio/{filename}.
    """
    if language is None:
        language = "hi-IN" if _is_mostly_hindi(text) else DEFAULT_LANGUAGE
    if speaker is None:
        speaker = DEFAULT_SPEAKER

    payload = {
        "text": text,
        "target_language_code": language,
        "speaker": speaker,
        "model": "bulbul:v3",
        "output_audio_codec": "mp3",
        "pace": 1.0,
    }

    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY is not set in .env")

    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }

    response = requests.post(TTS_URL, json=payload, headers=headers, timeout=30)

    if not response.ok:
        try:
            err_body = response.json()
        except Exception:
            err_body = response.text or f"HTTP {response.status_code}"
        log.error("Sarvam TTS failed: status=%s body=%s", response.status_code, err_body)
        if response.status_code == 403:
            raise ValueError(
                "Sarvam API 403: Invalid key, expired key, or no quota. "
                "Check https://dashboard.sarvam.ai/ → API Keys and usage. "
                f"Response: {err_body}"
            )
        response.raise_for_status()

    data = response.json()
    audios = data.get("audios") or []
    if not audios:
        raise ValueError("Sarvam TTS returned no audio")

    audio_bytes = base64.b64decode(audios[0])
    filename = f"{uuid.uuid4()}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(audio_bytes)

    return filename
