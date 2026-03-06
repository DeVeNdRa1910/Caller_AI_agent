"""
main.py — Ultra-low-latency voice pipeline for Sharma Logistics SONY bot.
         Voice provider: Plivo (replaces Twilio)

LATENCY BUDGET (user stops speaking → agent audio starts):
  speechTimeout      0.8s   fixed (Plivo GetInput)
  Plivo POST         0.2s   fixed (network)
  LLM + TTS overlap  0.6-1.0s  ← KEY optimization
  Plivo fetches WAV  0.1s   mulaw file is ~6x smaller than MP3
  Plivo play start   0.1s
  ─────────────────────────────────────────────
  Total target:      ~2.0-3.0s  (cache hit: ~1.5s, cache miss: ~2.5-3.5s)

PLIVO MIGRATION NOTES (Twilio → Plivo):
  - twilio SDK              → plivo SDK
  - TwiML <Gather>          → Plivo XML <GetInput>
  - TwiML <Play>            → Plivo XML <Play>
  - CallSid form param      → CallUUID form param
  - SpeechResult form param → speech form param
  - Confidence form param   → confidence form param
  - Direction values        → "inbound" / "outbound-api"
  - twilio.rest.Client()    → plivo.RestClient()
  - calls.create(twiml=...) → calls.create(answer_url=..., answer_method=...)
  - status_callback         → hangup_url (Plivo fires on call end)
  - Fallback URL            → fallback_url param in calls.create()
  - Audio MIME for Plivo    → audio/x-wav (mulaw WAV) works natively

ENV VARS REQUIRED:
  PLIVO_AUTH_ID        (replaces TWILIO_ACCOUNT_SID)
  PLIVO_AUTH_TOKEN     (replaces TWILIO_AUTH_TOKEN)
  PLIVO_PHONE_NUMBER   (replaces TWILIO_PHONE_NUMBER)
  NGROK_URL
  GROQ_API_KEY
  SARVAM_API_KEY
  GROQ_MODEL           (optional, default: llama-3.1-8b-instant)
"""

import logging
import os
import asyncio
import re
import time
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response, JSONResponse
from groq import AsyncGroq
import plivo                          # pip install plivo
from tts import (
    generate_tts_async,
    synthesise_text,
    llm_to_tts_stream,
    close_http_client,
    warmup_ws,
    AUDIO_DIR,
    TTS_FILE_EXT,
)
from media_stream import handle_media_stream

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

app         = FastAPI()
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

_call_history: dict[str, list[dict[str, str]]] = {}

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ─────────────────────────────────────────────────────────────────────────────
# PRE-RENDERED RESPONSE CACHE
# ─────────────────────────────────────────────────────────────────────────────
FIXED_RESPONSES: dict[str, str] = {
    "step1_hi":   "आगे बढ़ने से पहले, आप किस भाषा में बात करना पसंद करेंगे — हिंदी या इंग्लिश?",
    "step1_en":   "Before we proceed, which language would you prefer — Hindi or English?",
    "step2_hi":   "क्या हमारी ब्रांच से किसी ने आपको पहले कॉल किया है और कोटेशन भेजा है?",
    "step2_en":   "Has anyone from our branch already called you and shared a quotation?",
    "step3_hi":   "आप एक BHK, दो BHK या तीन BHK शिफ्ट कर रहे हैं?",
    "step3_en":   "Are you shifting a 1 BHK, 2 BHK, or 3 BHK household?",
    "step4_hi":   "पिकअप का फ्लोर नंबर क्या है? लिफ्ट है या नहीं? और क्या कोई गाड़ी शिफ्ट करनी है?",
    "step4_en":   "What is the pickup floor number? Is there a lift? Are any vehicles being shifted?",
    "step5_hi":   "आप कोटेशन ईमेल पर चाहेंगे या व्हाट्सऐप पर?",
    "step5_en":   "Would you like the quotation on email or WhatsApp?",
    "step6_hi":   "कृपया पूरा पिकअप पता पिनकोड सहित बताएं।",
    "step6_en":   "Please share the complete pickup address with pincode.",
    "step7_hi":   "किस दिन और समय पर सर्वे के लिए सुविधाजनक रहेगा?",
    "step7_en":   "Which day and time would be convenient for the survey?",
    "close_hi":   "आपका समय देने के लिए धन्यवाद। मैंने आपकी जानकारी नोट कर ली है और सर्वे शेड्यूल कर दिया है। आपका दिन शुभ रहे।",
    "close_en":   "Thank you for your time. I have noted your details and scheduled the survey. Have a great day.",
    "unclear_hi": "कृपया दोबारा बोलें।",
    "unclear_en": "Could you please repeat that?",
}

# text.strip() → audio filename (on disk)
_audio_cache: dict[str, str] = {}

OUTBOUND_OPENING = (
    "नमस्ते। मैं सोनी बोल रही हूं, शर्मा लॉजिस्टिक्स की तरफ से। "
    "मैं आपके घर शिफ्टिंग की इन्क्वायरी के संबंध में कॉल कर रही हूं। "
    "क्या अभी आप एक मिनट बात कर सकते हैं?"
)
_opening_audio: str | None = None

SYSTEM_PROMPT = """CRITICAL LANGUAGE POLICY (MANDATORY – OVERRIDES ALL OTHER INSTRUCTIONS):
1. You MUST always reply in the same language as the user's most recent message.
2. If the user speaks Hindi, respond completely in Hindi.
3. If the user speaks English, respond completely in English.
4. If the user speaks in mixed Hindi and English (Hinglish), respond in simple Hinglish.
5. Never default to English automatically.
6. If you are unsure about the user's preferred language, ask:
   "आप किस भाषा में बात करना चाहेंगे — हिंदी या इंग्लिश?"
7. Once the user selects a language, continue the entire conversation strictly in that language unless the user switches.
8. Keep all responses short (maximum 2–3 sentences) and suitable for a phone conversation.

------------------------------------------------------------

You are SONY, a polite and professional AI voice assistant representing Sharma Logistics.

Your goal is to:
- Qualify a household shifting enquiry
- Collect required details
- Build trust
- Schedule a free home survey

Always speak clearly, naturally, patiently, and respectfully.
Maintain a warm, helpful, and professional tone at all times.
Keep responses concise and conversational (no long paragraphs).

------------------------------------------------------------
CONVERSATION FLOW

OPENING (For inbound calls):
नमस्ते। मैं सोनी बोल रही हूं, शर्मा लॉजिस्टिक्स की तरफ से। मैं आपके घर शिफ्टिंग की इन्क्वायरी में मदद करने वाली AI सहायक हूं। क्या अभी आप एक मिनट बात कर सकते हैं?

OPENING (For outbound calls):
नमस्ते। मैं सोनी बोल रही हूं, शर्मा लॉजिस्टिक्स की तरफ से। मैं आपके घर शिफ्टिंग की इन्क्वायरी के संबंध में कॉल कर रही हूं। क्या अभी आप एक मिनट बात कर सकते हैं?

If the user says it is NOT a good time:
कोई बात नहीं। जब भी सुविधा हो कृपया कॉल कर लें। धन्यवाद।
(Politely end the call.)

If user agrees to talk:
आगे बढ़ने से पहले, आप किस भाषा में बात करना पसंद करेंगे — हिंदी या इंग्लिश?

------------------------------------------------------------
PURPOSE OF THE CALL

If Hindi selected:
धन्यवाद। मैं आपकी इन्दौर, मध्य प्रदेश से पुणे, महाराष्ट्र तक घर का सामान शिफ्ट करने की इन्क्वायरी के बारे में कॉल कर रही हूं। बस कुछ बातें पक्की कर लूं ताकि हम ठीक से मदद कर सकें।

If English selected:
Thank you. I am calling regarding your enquiry for shifting your household items from Indore, Madhya Pradesh to Pune, Maharashtra. I just need to confirm a few details so that we can assist you properly.

------------------------------------------------------------
QUESTIONS FLOW (Ask one at a time, wait for response)

Q1 – Branch Contact Status
Hindi: क्या हमारी ब्रांच से किसी ने आपको पहले कॉल किया है और कोटेशन भेजा है?
English: Has anyone from our branch already called you and shared a quotation?

If NO:
Hindi: देरी के लिए माफी चाहती हूं। हम तुरंत आपकी मदद करेंगे।
English: I sincerely apologize for the delay. We will assist you immediately.

------------------------------------------------------------

Q2 – Household Size
Hindi: आप एक BHK, दो BHK या तीन BHK शिफ्ट कर रहे हैं?
English: Are you shifting a 1 BHK, 2 BHK, or 3 BHK household?

------------------------------------------------------------

Q3 – Move Details
Hindi: पिकअप का फ्लोर नंबर क्या है? लिफ्ट है या नहीं? और क्या कोई गाड़ी शिफ्ट करनी है?
English: What is the pickup floor number? Is there a lift? Are any vehicles being shifted?

------------------------------------------------------------

Q4 – Quotation Preference
Hindi: आप कोटेशन ईमेल पर चाहेंगे या व्हाट्सऐप पर?
English: Would you like the quotation on email or WhatsApp?

------------------------------------------------------------

Q5 – Address Collection
Hindi: कृपया पूरा पिकअप पता पिनकोड सहित बताएं।
English: Please share the complete pickup address with pincode.

------------------------------------------------------------

Q6 – Survey Scheduling
Hindi: किस दिन और समय पर सर्वे के लिए सुविधाजनक रहेगा?
English: Which day and time would be convenient for the survey?

------------------------------------------------------------

TRUST BUILDING STATEMENT

Hindi:
सर्वे के दौरान हमारा फील्ड ऑफिसर सुरक्षित पैकिंग, इंश्योरेंस विकल्प और पारदर्शी कोटेशन की पूरी जानकारी देगा। कोई छुपा चार्ज नहीं होगा।

English:
During the survey, our field officer will explain safe packing, insurance options, and provide a transparent quotation with no hidden charges.

------------------------------------------------------------

If user is concerned about price:

Hindi:
आप सिर्फ उतने सामान का भुगतान करेंगे जितना आप शिफ्ट करवाते हैं। अंतिम कोटेशन सामान और दूरी के अनुसार होगा।

English:
You only pay for the items you move. The final quotation depends on the items and distance.

------------------------------------------------------------

CLOSING

Hindi:
आपका समय देने के लिए धन्यवाद। मैंने आपकी जानकारी नोट कर ली है और सर्वे शेड्यूल कर दिया है। आपका दिन शुभ रहे।

English:
Thank you for your time. I have noted your details and scheduled the survey. Have a great day.
"""


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _is_inbound(direction: str | None) -> bool:
    """
    Plivo sets Direction to 'inbound' for incoming calls
    and 'outbound-api' for API-initiated calls.
    """
    if not direction:
        return True
    return direction.strip().lower() == "inbound"


def _form_get(form_like, key: str, default=None):
    """Case-insensitive form field lookup."""
    v = form_like.get(key)
    if v is not None and str(v).strip() != "":
        return v
    if key != key.lower():
        v = form_like.get(key.lower())
        if v is not None and str(v).strip() != "":
            return v
    return default


def _cache_get(text: str) -> str | None:
    filename = _audio_cache.get(text.strip())
    if filename and os.path.isfile(os.path.join(AUDIO_DIR, filename)):
        return filename
    return None


def _cache_consume(text: str) -> str | None:
    filename = _cache_get(text)
    if filename:
        _audio_cache.pop(text.strip(), None)
        asyncio.create_task(_rerender_bg(text))
    return filename


async def _rerender_bg(text: str):
    try:
        filename = await synthesise_text(text)
        _audio_cache[text.strip()] = filename
        log.debug("Cache refilled: %s", text[:40])
    except Exception as e:
        log.warning("Cache re-render failed: %s", e)


async def _prerender_all():
    """Render all fixed responses CONCURRENTLY using the 2-connection WS pool."""
    async def _one(key: str, text: str):
        try:
            fn = await synthesise_text(text)
            _audio_cache[text.strip()] = fn
            log.info("  ✅ [%s] cached", key)
        except Exception as e:
            log.warning("  ❌ [%s] failed: %s", key, e)

    await asyncio.gather(*[_one(k, v) for k, v in FIXED_RESPONSES.items()])


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

async def _run_pipeline(
    messages: list[dict], call_uuid: str, user_input: str
) -> tuple[list[str], str]:
    """
    TRUE parallel LLM+TTS:
      1. Open Groq stream
      2. Immediately pass stream to llm_to_tts_stream()
      3. Return single audio file
    """
    t0 = time.perf_counter()

    stream = await groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=80,
        temperature=0.1,
        stream=True,
    )
    log.info("⏱  Groq stream open: %.3fs", time.perf_counter() - t0)

    filename, ai_reply = await llm_to_tts_stream(stream)
    ai_reply = ai_reply.strip() or "कृपया दोबारा बोलें।"
    log.info("⏱  LLM+TTS done: %.3fs | %s", time.perf_counter() - t0, ai_reply[:60])

    if call_uuid:
        hist = _call_history.setdefault(call_uuid, [])
        hist.append({"role": "user",      "content": user_input})
        hist.append({"role": "assistant", "content": ai_reply})

    log.info("⏱  pipeline total: %.3fs", time.perf_counter() - t0)
    return [filename], ai_reply


# ─────────────────────────────────────────────────────────────────────────────
# PLIVO XML BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_response_xml(
    ngrok_url: str,
    audio_filenames: list[str],
    action_url: str,
) -> str:
    """
    Build Plivo XML response.

    Plivo equivalents:
      <Play>    — plays an audio URL (same name as TwiML)
      <GetInput>— replaces TwiML <Gather>; collects speech/DTMF
        - inputType="speech"   → speech recognition
        - action               → webhook URL for result POST
        - method               → POST
        - speechEndTimeout     → replaces speechTimeout (seconds of silence)
        - timeout              → max wait before no-input
        - language             → BCP-47 language code
        - finishOnKey          → (omitted; speech-only)

    NOTE: Plivo posts the recognised text as the 'speech' param (not SpeechResult).
    """
    play_elements = "".join(
        f"<Play>{ngrok_url}/audio/{fn}</Play>" for fn in audio_filenames
    )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    {play_elements}
    <GetInput
        inputType="speech"
        action="{action_url}"
        method="POST"
        speechEndTimeout="0.8"
        timeout="5"
        language="hi-IN"
        redirect="true">
    </GetInput>
</Response>"""


def _build_error_xml(ngrok_url: str) -> str:
    """Plivo XML fallback on internal error."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Speak language="hi-IN">Technical issue. Please try again.</Speak>
    <GetInput
        inputType="speech"
        action="{ngrok_url}/voice"
        method="POST"
        speechEndTimeout="0.8"
        timeout="5"
        language="hi-IN"
        redirect="true">
    </GetInput>
</Response>"""


# ─────────────────────────────────────────────────────────────────────────────
# CORE VOICE HANDLER
# ─────────────────────────────────────────────────────────────────────────────

async def _handle_voice(form_like) -> Response:
    t0        = time.perf_counter()
    ngrok_url = os.getenv("NGROK_URL", "").rstrip("/")

    # ── Plivo param names ──────────────────────────────────────────────────
    # CallUUID  → unique call identifier  (was CallSid in Twilio)
    # speech    → recognised speech text  (was SpeechResult in Twilio)
    # confidence→ speech confidence score (same concept, different casing)
    # Direction → "inbound" or "outbound-api"
    direction = _form_get(form_like, "Direction") or ""
    call_uuid = (_form_get(form_like, "CallUUID") or "").strip()
    speech    = _form_get(form_like, "speech")          # Plivo posts as 'speech'
    conf      = _form_get(form_like, "confidence")

    log.info("▶ turn: uuid=%s dir=%s conf=%s speech=%s",
             call_uuid, direction, conf, (speech or "")[:80] or "<empty>")

    try:
        history = _call_history.get(call_uuid, [])
        is_cont = len(history) > 0

        if not speech or not speech.strip():
            if is_cont:
                user_input = "[unclear]"
            elif _is_inbound(direction):
                user_input = "[inbound_start]"
            else:
                user_input = "[outbound_start]"
        else:
            user_input = speech.strip()

        audio_filenames: list[str] = []
        ai_reply = None

        # Fast path: unclear speech → pre-rendered "please repeat"
        if user_input == "[unclear]":
            for key in ("unclear_hi", "unclear_en"):
                text = FIXED_RESPONSES[key]
                fn   = _cache_consume(text)
                if fn:
                    audio_filenames = [fn]
                    ai_reply        = text
                    log.info("⚡ Unclear fast path from cache")
                    break

        if not audio_filenames:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for msg in history[-6:]:
                messages.append(msg)
            messages.append({"role": "user", "content": user_input})
            audio_filenames, ai_reply = await _run_pipeline(messages, call_uuid, user_input)

        action_url = f"{ngrok_url}/voice"
        twiml      = _build_response_xml(ngrok_url, audio_filenames, action_url)

        log.info("⏱  handler total: %.3fs", time.perf_counter() - t0)
        return Response(content=twiml, media_type="text/xml")

    except Exception as e:
        log.exception("Voice handler error: %s", e)
        return Response(content=_build_error_xml(ngrok_url), media_type="text/xml")


# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE
# ─────────────────────────────────────────────────────────────────────────────

def _cleanup_stale_audio(max_age_seconds: int = 300):
    if not os.path.isdir(AUDIO_DIR):
        return
    now     = time.time()
    removed = 0
    for name in os.listdir(AUDIO_DIR):
        path = os.path.join(AUDIO_DIR, name)
        try:
            if os.path.isfile(path) and (now - os.path.getmtime(path)) > max_age_seconds:
                os.remove(path)
                removed += 1
        except OSError:
            pass
    if removed:
        log.info("Cleaned %d stale audio file(s)", removed)


async def _cleanup_audio_loop(interval: int = 120, max_age: int = 300):
    while True:
        await asyncio.sleep(interval)
        _cleanup_stale_audio(max_age)


@app.on_event("startup")
async def startup():
    global _opening_audio
    log.info("Starting up — establishing WS pool + pre-rendering responses…")

    await warmup_ws()

    _opening_audio = await synthesise_text(OUTBOUND_OPENING)
    log.info("✅ Opening audio ready: %s", _opening_audio)

    async def _bg():
        await _prerender_all()
        log.info("✅ Fixed response cache warm. %d/%d entries",
                 len(_audio_cache), len(FIXED_RESPONSES))

    asyncio.create_task(_bg())
    asyncio.create_task(_cleanup_audio_loop())
    log.info("✅ Startup done — server ready.")


@app.on_event("shutdown")
async def shutdown():
    try:
        from stt import close_stt_client
        await close_stt_client()
    except Exception:
        pass
    await close_http_client()
    log.info("Shutdown complete.")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "cache_entries": len(_audio_cache)}


@app.post("/voice")
async def voice_webhook(request: Request):
    """
    Main inbound webhook for Plivo.
    Set this URL as the 'Answer URL' in your Plivo application/number settings.
    Plivo will POST: CallUUID, Direction, From, To, speech (after GetInput), confidence, etc.
    """
    form   = await request.form()
    params = dict(request.query_params)
    return await _handle_voice({**params, **dict(form)})


@app.post("/voice-stream")
async def voice_stream_webhook(request: Request):
    """
    Plivo XML to connect call to a media stream WebSocket.
    Plivo uses <Stream> inside <Connect> — same concept as Twilio.
    """
    ngrok_url  = os.getenv("NGROK_URL", "").rstrip("/")
    stream_ws  = ngrok_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    if not stream_ws.endswith("/media-stream"):
        stream_ws = stream_ws.rstrip("/") + "/media-stream"

    # Plivo <Stream> XML (same structure as Twilio)
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream keepCallAlive="true" bidirectional="true" contentType="audio/x-mulaw;rate=8000" url="{stream_ws}" />
    </Connect>
</Response>"""
    return Response(content=twiml, media_type="text/xml")


@app.websocket("/media-stream")
async def media_stream_ws(websocket: WebSocket):
    await handle_media_stream(websocket, SYSTEM_PROMPT, _call_history, groq_client)


@app.post("/voice/fallback")
async def voice_fallback():
    """
    Plivo fallback URL — set as 'Fallback Answer URL' in Plivo app settings.
    Called when the primary answer URL fails or times out.
    """
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Speak language="hi-IN">Could not connect. Goodbye.</Speak>
    <Hangup/>
</Response>"""
    return Response(content=twiml, media_type="text/xml")


@app.post("/voice/status")
async def voice_status(request: Request):
    """
    Plivo call status / hangup callback.
    Set as 'Hangup URL' in your Plivo application.

    Plivo posts: CallUUID, CallStatus, Duration, etc.
    CallStatus values: 'completed', 'busy', 'failed', 'no-answer', 'canceled'
    """
    try:
        body      = dict(await request.form())
        call_uuid = body.get("CallUUID")                     # Plivo: CallUUID
        status    = (body.get("CallStatus") or "").strip().lower()
        log.info("Call status: %s → %s", call_uuid, status)
        if call_uuid and status == "completed":
            _call_history.pop(call_uuid, None)
    except Exception:
        pass
    return Response(content="", status_code=200)


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """
    Serve TTS audio for Plivo <Play>.
    Returns audio/x-wav (mulaw 8kHz) — Plivo plays this natively.
    File is deleted after serving to prevent accumulation.
    """
    if not (filename.endswith(".wav") or filename.endswith(".mp3")):
        return Response(status_code=404)
    if ".." in filename or "/" in filename:
        return Response(status_code=404)
    filepath = os.path.join(AUDIO_DIR, filename)
    if not os.path.isfile(filepath):
        return Response(status_code=404)
    with open(filepath, "rb") as f:
        data = f.read()
    try:
        os.remove(filepath)
        log.debug("Deleted after serve: %s", filename)
    except OSError as e:
        log.warning("Could not delete %s: %s", filename, e)

    # Plivo accepts mulaw WAV as audio/x-wav or audio/wav.
    # audio/basic (raw mulaw) also works but audio/x-wav is more explicit.
    mime = "audio/x-wav" if filename.endswith(".wav") else "audio/mpeg"
    return Response(content=data, media_type=mime)


# ─────────────────────────────────────────────────────────────────────────────
# OUTBOUND CALL  (Plivo SDK)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/call-user")
async def call_user(mobile_number: str):
    """
    Initiate an outbound call via Plivo REST API.

    Plivo differences vs Twilio:
      - plivo.RestClient(auth_id, auth_token)  instead of twilio.Client(sid, token)
      - calls.create(from_=..., to_=..., answer_url=..., answer_method=...)
        instead of calls.create(from_=..., to=..., twiml=...)
      - Plivo answer_url must return Plivo XML (not inline TwiML string)
      - hangup_url replaces status_callback
      - hangup_url_method replaces status_callback_method
      - No 'status_callback_event' filter — Plivo always calls hangup_url on completion

    IMPORTANT: The opening audio is played by the answer_url endpoint (/voice-outbound-open),
    not inline in the API call (Plivo doesn't accept inline TwiML like Twilio does).
    """
    global _opening_audio

    auth_id       = os.getenv("PLIVO_AUTH_ID")
    auth_token    = os.getenv("PLIVO_AUTH_TOKEN")
    plivo_number  = os.getenv("PLIVO_PHONE_NUMBER")
    ngrok_url     = os.getenv("NGROK_URL", "").rstrip("/")

    if not ngrok_url:
        return JSONResponse({"error": "NGROK_URL not set"}, status_code=400)
    if not auth_id or not auth_token or not plivo_number:
        return JSONResponse({"error": "Plivo credentials not set in .env"}, status_code=400)

    # Prepare opening audio file
    if _opening_audio and os.path.isfile(os.path.join(AUDIO_DIR, _opening_audio)):
        audio_filename = _opening_audio
        _opening_audio = None
    else:
        try:
            audio_filename = await synthesise_text(OUTBOUND_OPENING)
        except Exception as e:
            return JSONResponse({"error": f"TTS failed: {e}"}, status_code=500)

    # Store the audio filename temporarily so /voice-outbound-open can serve it.
    # A real production system would use a short-lived token or pass via query param.
    _pending_outbound_audio[mobile_number] = audio_filename

    # Plivo answer_url — called when callee picks up
    answer_url = f"{ngrok_url}/voice-outbound-open?to={mobile_number}"

    plivo_client = plivo.RestClient(auth_id, auth_token)

    try:
        response = plivo_client.calls.create(
            from_         = plivo_number,
            to_           = mobile_number,
            answer_url    = answer_url,
            answer_method = "GET",                         # Plivo fetches XML via GET by default
            hangup_url    = f"{ngrok_url}/voice/status",
            hangup_method = "POST",
            fallback_url  = f"{ngrok_url}/voice/fallback",
            fallback_method = "POST",
        )
        call_uuid = response[1].get("request_uuid") or str(response)
        log.info("Outbound call initiated: %s", call_uuid)
        return {"status": "calling", "request_uuid": call_uuid}
    except Exception as e:
        log.exception("Plivo call creation failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


# Temporary store for outbound opening audio filenames keyed by destination number
_pending_outbound_audio: dict[str, str] = {}


@app.get("/voice-outbound-open")
async def voice_outbound_open(to: str = ""):
    """
    Plivo fetches this URL when the outbound call is answered.
    Plays the opening audio then hands off to the main /voice webhook
    via <GetInput> action.

    NOTE: Plivo answer_url is fetched with GET by default (answer_method="GET").
    The 'to' query param is used to look up the pre-rendered opening audio.
    """
    ngrok_url      = os.getenv("NGROK_URL", "").rstrip("/")
    audio_filename = _pending_outbound_audio.pop(to, None)

    if not audio_filename or not os.path.isfile(os.path.join(AUDIO_DIR, audio_filename)):
        # Fallback: re-synthesise if pre-rendered file was lost
        try:
            audio_filename = await synthesise_text(OUTBOUND_OPENING)
        except Exception as e:
            log.error("Failed to synthesise outbound opening: %s", e)
            return Response(
                content="""<?xml version="1.0" encoding="UTF-8"?>
<Response><Speak language="hi-IN">नमस्ते, कृपया प्रतीक्षा करें।</Speak></Response>""",
                media_type="text/xml",
            )

    audio_url  = f"{ngrok_url}/audio/{audio_filename}"
    action_url = f"{ngrok_url}/voice"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Play>{audio_url}</Play>
    <GetInput
        inputType="speech"
        action="{action_url}"
        method="POST"
        speechEndTimeout="0.8"
        timeout="5"
        language="hi-IN"
        redirect="true">
    </GetInput>
</Response>"""
    return Response(content=twiml, media_type="text/xml")