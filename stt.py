import os
import requests
from dotenv import load_dotenv
load_dotenv()

SARVAM_API_KEY = (os.getenv("SARVAM_API_KEY") or "").strip()
STT_URL = "https://api.sarvam.ai/speech-to-text"


def transcribe_audio(
    audio_path: str | None = None,
    audio_bytes: bytes | None = None,
    mode: str = "codemix",
    content_type: str = "audio/wav",
    filename: str = "audio.wav",
) -> str:
    """
    Transcribe audio using Sarvam AI STT (Saaras v3).
    Provide either audio_path (path to file) or audio_bytes.
    mode: "codemix" (best for Hindi/English mix), "transcribe", "translate", "verbatim", "translit"
    content_type/filename: use audio/mpeg + "audio.mp3" when passing MP3 bytes.
    Returns the transcript text.
    """
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY is not set")

    headers = {"api-subscription-key": SARVAM_API_KEY}

    if audio_path:
        with open(audio_path, "rb") as f:
            name = os.path.basename(audio_path)
            files = {"file": (name, f, content_type)}
            data = {"model": "saaras:v3", "mode": mode}
            response = requests.post(STT_URL, headers=headers, data=data, files=files, timeout=30)
    elif audio_bytes:
        files = {"file": (filename, audio_bytes, content_type)}
        data = {"model": "saaras:v3", "mode": mode}
        response = requests.post(STT_URL, headers=headers, data=data, files=files, timeout=30)
    else:
        raise ValueError("Provide either audio_path or audio_bytes")

    response.raise_for_status()
    result = response.json()
    return (result.get("transcript") or "").strip() or ""
