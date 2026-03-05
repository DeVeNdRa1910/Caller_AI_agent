"""
main.py — Ultra-low-latency voice pipeline for Sharma Logistics SONY bot.

LATENCY BUDGET (user stops speaking → agent audio starts):
  speechTimeout      0.8s   fixed (Twilio)
  Twilio POST        0.2s   fixed (network)
  LLM + TTS overlap  0.6-1.0s  ← KEY optimization
  Twilio fetches WAV 0.1s   mulaw file is ~6x smaller than MP3
  Twilio play start  0.1s
  ─────────────────────────────────────────────
  Total target:      ~2.0-3.0s  (cache hit: ~1.5s, cache miss: ~2.5-3.5s)

KEY CHANGES vs previous version:
  1. LLM+TTS TRUE PARALLEL: Start Groq stream, immediately pass stream object
     to llm_to_tts_stream(). LLM first tokens go to Sarvam WS while LLM is
     still generating. No full-buffer-then-TTS pattern.

  2. CACHE CHECK BEFORE LLM: If LLM would produce a fixed phrase, we already
     have it cached. Check cache after LLM (not before — we don't know the
     reply yet) but this is handled in tts.py synthesise() automatically.

  3. MULAW MIME TYPE: serve_audio returns audio/basic (mulaw MIME) so Twilio
     recognises it instantly without probing. No MP3 transcode delay.

  4. HISTORY TRIMMED TO LAST 6 MESSAGES (3 turns): Reduces LLM context size
     → faster TTFT. This bot asks short sequential questions; 3 turns of
     history is enough context.

  5. PARALLEL PRE-RENDER: All fixed phrases render concurrently at startup
     using both WS pool connections simultaneously.
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
from twilio.rest import Client
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

    # Concurrent pre-render — 2 WS connections handle 2 at a time
    await asyncio.gather(*[_one(k, v) for k, v in FIXED_RESPONSES.items()])


async def _run_pipeline(
    messages: list[dict], call_sid: str, user_input: str
) -> tuple[list[str], str]:
    """
    TRUE parallel LLM+TTS:
      1. Open Groq stream
      2. Immediately pass stream to llm_to_tts_stream()
         → LLM tokens flow directly to Sarvam WS as they arrive
         → No buffering the full reply before starting TTS
      3. Return single audio file (whole reply in one WS request)

    This is faster than sentence-by-sentence parallel REST because:
      - No sentence-splitting delay
      - Single WS flush = single audio file = single Twilio <Play>
      - Sarvam WS starts synthesising on first token (~100ms into LLM)
    """
    t0 = time.perf_counter()

    stream = await groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=80,      # hard cap keeps replies short and TTS fast
        temperature=0.1,
        stream=True,
    )
    log.info("⏱  Groq stream open: %.3fs", time.perf_counter() - t0)

    # llm_to_tts_stream() runs LLM drain and WS send/recv concurrently
    filename, ai_reply = await llm_to_tts_stream(stream)
    ai_reply = ai_reply.strip() or "कृपया दोबारा बोलें।"
    log.info("⏱  LLM+TTS done: %.3fs | %s", time.perf_counter() - t0, ai_reply[:60])

    if call_sid:
        hist = _call_history.setdefault(call_sid, [])
        hist.append({"role": "user",      "content": user_input})
        hist.append({"role": "assistant", "content": ai_reply})

    log.info("⏱  pipeline total: %.3fs", time.perf_counter() - t0)
    return [filename], ai_reply


async def _handle_voice(form_like) -> Response:
    t0        = time.perf_counter()
    ngrok_url = os.getenv("NGROK_URL", "").rstrip("/")
    direction = _form_get(form_like, "Direction") or ""
    call_sid  = (_form_get(form_like, "CallSid") or "").strip()
    speech    = _form_get(form_like, "SpeechResult")
    conf      = _form_get(form_like, "Confidence")

    log.info("▶ turn: sid=%s dir=%s conf=%s speech=%s",
             call_sid, direction, conf, (speech or "")[:80] or "<empty>")

    try:
        history = _call_history.get(call_sid, [])
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
            # Last 6 messages = 3 turns of context. Enough for this linear flow.
            # Fewer messages = smaller LLM context = faster TTFT.
            for msg in history[-6:]:
                messages.append(msg)
            messages.append({"role": "user", "content": user_input})
            audio_filenames, ai_reply = await _run_pipeline(messages, call_sid, user_input)

        record_action = f"{ngrok_url}/voice"
        play_elements = "".join(
            f'<Play>{ngrok_url}/audio/{fn}</Play>' for fn in audio_filenames
        )
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    {play_elements}
    <Gather input="speech" action="{record_action}" method="POST"
            speechTimeout="0.8" timeout="5" actionOnEmptyResult="true"
            language="hi-IN" enhanced="true" />
</Response>"""
        log.info("⏱  handler total: %.3fs", time.perf_counter() - t0)
        return Response(content=twiml, media_type="text/xml")

    except Exception as e:
        log.exception("Voice handler error: %s", e)
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="hi-IN">Technical issue. Please try again.</Say>
    <Gather input="speech" action="{ngrok_url}/voice" method="POST"
            speechTimeout="0.8" timeout="5" actionOnEmptyResult="true" language="hi-IN" />
</Response>"""
        return Response(content=twiml, media_type="text/xml")


# ── Lifecycle ──────────────────────────────────────────────────────────────────

def _cleanup_stale_audio(max_age_seconds: int = 300):
    if not os.path.isdir(AUDIO_DIR):
        return
    now = time.time()
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

    # Step 1: Connect both WS pool connections concurrently
    await warmup_ws()

    # Step 2: Render opening audio (blocks until ready so /call-user works immediately)
    _opening_audio = await synthesise_text(OUTBOUND_OPENING)
    log.info("✅ Opening audio ready: %s", _opening_audio)

    # Step 3: Pre-render all fixed responses concurrently in background
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


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "cache_entries": len(_audio_cache)}


@app.post("/voice")
async def voice_webhook(request: Request):
    form   = await request.form()
    params = dict(request.query_params)
    return await _handle_voice({**params, **dict(form)})


@app.post("/voice-stream")
async def voice_stream_webhook(request: Request):
    ngrok_url = os.getenv("NGROK_URL", "").rstrip("/")
    stream_ws = ngrok_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    if not stream_ws.endswith("/media-stream"):
        stream_ws = stream_ws.rstrip("/") + "/media-stream"
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{stream_ws}" />
    </Connect>
</Response>"""
    return Response(content=twiml, media_type="text/xml")


@app.websocket("/media-stream")
async def media_stream_ws(websocket: WebSocket):
    await handle_media_stream(websocket, SYSTEM_PROMPT, _call_history, groq_client)


@app.post("/voice/fallback")
async def voice_fallback():
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="hi-IN">Could not connect. Goodbye.</Say>
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
    """
    Serve TTS audio for Twilio <Play>.
    Returns audio/basic (mulaw) — Twilio plays this natively with no transcoding.
    File is deleted after serving to prevent accumulation.
    """
    # Accept both .wav (mulaw) and .mp3 (fallback legacy files)
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

    # mulaw 8kHz → audio/basic. MP3 legacy → audio/mpeg.
    mime = "audio/basic" if filename.endswith(".wav") else "audio/mpeg"
    return Response(content=data, media_type=mime)


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

    if _opening_audio and os.path.isfile(os.path.join(AUDIO_DIR, _opening_audio)):
        audio_filename = _opening_audio
        _opening_audio = None
    else:
        try:
            audio_filename = await synthesise_text(OUTBOUND_OPENING)
        except Exception as e:
            return JSONResponse({"error": f"TTS failed: {e}"}, status_code=500)

    audio_url = f"{ngrok_url}/audio/{audio_filename}"

    stream_ws = ngrok_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    if not stream_ws.endswith("/media-stream"):
        stream_ws = stream_ws.rstrip("/") + "/media-stream"

    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"<Play>{audio_url}</Play>"
        f'<Connect><Stream url="{stream_ws}" /></Connect>'
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