import logging
import os
import asyncio
import time
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse
from groq import AsyncGroq
from twilio.rest import Client
from tts import (
    llm_to_tts_stream,
    generate_tts_async,
    close_http_client,
    warmup_ws,
    AUDIO_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

app = FastAPI()

groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

_call_history: dict[str, list[dict[str, str]]] = {}

# Fixed outbound opening — pre-rendered at startup, served instantly on every call
OUTBOUND_OPENING = (
    "नमस्ते। मैं सोनी बोल रही हूं, शर्मा लॉजिस्टिक्स की तरफ से। "
    "आपकी घर शिफ्टिंग इन्क्वायरी के संबंध में कॉल कर रही हूं। "
    "क्या अभी आप एक मिनट बात कर सकते हैं?"
)
_opening_audio: str | None = None   # path to pre-rendered MP3

SYSTEM_PROMPT = """You are SONY, a voice assistant for Sharma Logistics (phone call).

LANGUAGE RULES (STRICT):
- Default language: Hindi
- If user speaks English → reply in English only
- If user speaks Marathi → reply in Marathi only
- NEVER mix languages in one reply. Never use "/" to show both languages.
- One language per reply, always.

LENGTH RULE (CRITICAL):
- Maximum 1 sentence per reply. Hard limit.
- Short sentences only. This is a phone call.

CONVERSATION FLOW (ask questions one at a time, in order):

STEP 1 — Language check (only if user hasn't established language):
  Hindi: "आप किस भाषा में बात करना चाहेंगे — हिंदी या इंग्लिश?"

STEP 2 — Branch quotation:
  Hindi: "क्या हमारी ब्रांच से आपको कोटेशन मिला है?"
  English: "Has our branch given you a quotation?"

STEP 3 — BHK size:
  Hindi: "आप कितने BHK शिफ्ट कर रहे हैं?"
  English: "How many BHK are you shifting?"

STEP 4 — Floor and lift:
  Hindi: "पिकअप कौन से फ्लोर पर है और लिफ्ट है?"
  English: "Which floor is the pickup, and is there a lift?"

STEP 5 — Quotation channel:
  Hindi: "कोटेशन WhatsApp पर भेजूं या ईमेल पर?"
  English: "Should I send the quotation on WhatsApp or email?"

STEP 6 — Address:
  Hindi: "पिकअप का पूरा पता और पिनकोड बताएं।"
  English: "Please share the full pickup address with pincode."

STEP 7 — Survey time:
  Hindi: "सर्वे के लिए कौन सा दिन ठीक रहेगा?"
  English: "Which day works for the survey?"

CLOSING:
  Hindi: "धन्यवाद, सर्वे शेड्यूल हो गया है।"
  English: "Thank you, the survey has been scheduled."
"""


def _is_inbound(direction: str | None) -> bool:
    if not direction:
        return True
    return direction.strip().lower() == "inbound"


def _form_get(form_like, key: str, default=None):
    v = form_like.get(key)
    if v is not None and str(v).strip() != "":
        return v
    if key != key.lower():
        v = form_like.get(key.lower())
        if v is not None and str(v).strip() != "":
            return v
    return default


async def _run_pipeline(messages: list[dict], call_sid: str, user_input: str) -> str:
    """
    Streaming Groq → persistent Sarvam WS TTS → MP3 filename.

    No per-turn WS handshake (connection pre-opened at startup).
    LLM and TTS run in parallel — TTS starts at first sentence boundary.
    """
    t0 = time.perf_counter()

    groq_stream = await groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=60,     # 1 Hindi sentence ≈ 40-55 tokens; hard cap keeps audio short
        temperature=0.3,
        stream=True,
    )
    log.info("⏱  Groq stream open: %.2fs", time.perf_counter() - t0)

    audio_filename, ai_reply = await llm_to_tts_stream(groq_stream)

    if not ai_reply:
        ai_reply = "कृपया दोबारा बोलें।"

    if call_sid:
        hist = _call_history.setdefault(call_sid, [])
        hist.append({"role": "user",      "content": user_input})
        hist.append({"role": "assistant", "content": ai_reply})

    log.info("⏱  _run_pipeline total: %.2fs", time.perf_counter() - t0)
    return audio_filename


async def _handle_voice(form_like) -> Response:
    t0         = time.perf_counter()
    ngrok_url  = os.getenv("NGROK_URL", "").rstrip("/")
    direction  = _form_get(form_like, "Direction") or ""
    call_sid   = (_form_get(form_like, "CallSid") or "").strip()
    user_input = _form_get(form_like, "SpeechResult")
    confidence = _form_get(form_like, "Confidence")

    log.info("▶ Voice turn: sid=%s dir=%s conf=%s speech=%s",
             call_sid, direction, confidence,
             (user_input or "")[:80] or "<empty>")

    try:
        history         = _call_history.get(call_sid, [])
        is_continuation = len(history) > 0

        if not user_input or not user_input.strip():
            if is_continuation:
                user_input = "[User spoke but audio was unclear. Ask them to repeat in one sentence.]"
            elif _is_inbound(direction):
                user_input = "[Inbound call just connected. Greet the user as SONY from Sharma Logistics.]"
            else:
                user_input = "[Outbound call just connected. Introduce yourself as SONY from Sharma Logistics and ask if they have a moment.]"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in history[-16:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_input})

        audio_filename = await _run_pipeline(messages, call_sid, user_input)

        audio_url     = f"{ngrok_url}/audio/{audio_filename}"
        record_action = f"{ngrok_url}/voice"

        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Play>{audio_url}</Play>
    <Gather input="speech" action="{record_action}" method="POST"
            speechTimeout="2" timeout="8" actionOnEmptyResult="true"
            language="hi-IN" enhanced="true" />
</Response>"""
        log.info("⏱  _handle_voice total: %.2fs", time.perf_counter() - t0)
        return Response(content=twiml, media_type="text/xml")

    except Exception as e:
        log.exception("Voice handler error: %s", e)
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="hi-IN">Technical issue. Please try again.</Say>
    <Gather input="speech" action="{ngrok_url}/voice" method="POST"
            speechTimeout="2" timeout="8" actionOnEmptyResult="true" language="hi-IN" />
</Response>"""
        return Response(content=twiml, media_type="text/xml")


# ── Lifecycle ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global _opening_audio
    log.info("Server starting — warming up WS + pre-rendering opening audio…")
    try:
        await warmup_ws()
        log.info("✅ WS TTS ready.")
    except Exception as e:
        log.warning("WS warmup failed (will retry on first call): %s", e)

    # Pre-render the fixed opening message so /call-user is instant
    try:
        _opening_audio = await generate_tts_async(OUTBOUND_OPENING, model="bulbul:v2", speaker="anushka")
        log.info("✅ Opening audio pre-rendered: %s", _opening_audio)
    except Exception as e:
        log.warning("Opening audio pre-render failed: %s", e)


@app.on_event("shutdown")
async def shutdown():
    await close_http_client()
    log.info("All connections closed.")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/voice")
async def voice_webhook(request: Request):
    """Single endpoint — no redirect overhead."""
    form   = await request.form()
    params = dict(request.query_params)
    return await _handle_voice({**params, **dict(form)})


@app.post("/voice/fallback")
async def voice_fallback():
    log.warning("Fallback hit")
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="hi-IN">Could not connect. Please try again. Goodbye.</Say>
    <Hangup/>
</Response>"""
    return Response(content=twiml, media_type="text/xml")


@app.post("/voice/status")
async def voice_status(request: Request):
    try:
        body     = dict(await request.form())
        call_sid = body.get("CallSid")
        status   = (body.get("CallStatus") or "").strip().lower()
        log.info("Call status: %s → %s", call_sid, status)
        if call_sid and status == "completed":
            _call_history.pop(call_sid, None)
    except Exception:
        pass
    return Response(content="", status_code=200)


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    if not filename.endswith(".mp3") or ".." in filename or "/" in filename:
        return Response(status_code=404)
    filepath = os.path.join(AUDIO_DIR, filename)
    if not os.path.isfile(filepath):
        log.warning("Audio not found: %s", filename)
        return Response(status_code=404)
    with open(filepath, "rb") as f:
        data = f.read()
    try:
        os.remove(filepath)
    except OSError:
        pass
    return Response(content=data, media_type="audio/mpeg")


# ── Outbound call ──────────────────────────────────────────────────────────────

@app.post("/call-user")
async def call_user(mobile_number: str):
    global _opening_audio
    account_sid   = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token    = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_number = os.getenv("TWILIO_PHONE_NUMBER")
    ngrok_url     = os.getenv("NGROK_URL", "").rstrip("/")

    if not ngrok_url:
        return JSONResponse({"error": "NGROK_URL not set"}, status_code=400)

    # Use pre-rendered opening audio if available (generated at startup = 0ms)
    # Fall back to generating fresh if the file was already served/deleted
    audio_filename = None
    if _opening_audio and os.path.isfile(os.path.join(AUDIO_DIR, _opening_audio)):
        audio_filename = _opening_audio
        _opening_audio = None   # consumed — will re-render next startup
        log.info("Using pre-rendered opening audio: %s", audio_filename)
    else:
        log.info("Pre-rendered audio not available — generating fresh…")
        try:
            audio_filename = await generate_tts_async(OUTBOUND_OPENING)
        except Exception as e:
            log.exception("Opening TTS failed: %s", e)
            return JSONResponse({"error": "TTS failed. Check SARVAM_API_KEY."}, status_code=500)

    audio_url     = f"{ngrok_url}/audio/{audio_filename}"
    record_action = f"{ngrok_url}/voice"
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"<Play>{audio_url}</Play>"
        f'<Gather input="speech" action="{record_action}" method="POST" '
        'speechTimeout="2" timeout="8" actionOnEmptyResult="true" '
        'language="hi-IN" enhanced="true" />'
        "</Response>"
    )

    twilio_client = Client(account_sid, auth_token)
    call = twilio_client.calls.create(
        to=mobile_number,
        from_=twilio_number,
        twiml=twiml,
        status_callback=f"{ngrok_url}/voice/status",
        status_callback_event=["completed"],
    )
    log.info("Outbound call initiated: %s", call.sid)
    return {"status": "calling", "call_sid": call.sid}