"""
main.py — Ultra-low-latency voice pipeline for Sharma Logistics SONY bot.

LATENCY STRATEGY:
  Pipeline (LLM+TTS): ~1.5-2.5s  ← already optimized
  Twilio overhead:    ~2-3s       ← speechTimeout(1s) + fetch + play start
  Total perceived:    ~3-5s target

KEY OPTIMIZATION: Pre-render all 7 fixed questions at startup.
On cache hit: 0ms LLM + 0ms TTS = ~200ms total response (just Twilio overhead).
On cache miss: normal LLM+TTS pipeline (~1.5-2s).
Cache auto-refills in background after each use.
"""

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

app        = FastAPI()
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

_call_history: dict[str, list[dict[str, str]]] = {}

# ─────────────────────────────────────────────────────────────────────────────
# PRE-RENDERED RESPONSE CACHE
# All 7 fixed questions + common fallbacks are rendered at startup.
# Cache entry: text → mp3 filename.  After serving, re-render in background.
# ─────────────────────────────────────────────────────────────────────────────
FIXED_RESPONSES: dict[str, str] = {
    "step1_hi":   "आप किस भाषा में बात करना चाहेंगे — हिंदी या इंग्लिश?",
    "step2_hi":   "क्या हमारी ब्रांच से आपको कोटेशन मिला है?",
    "step2_en":   "Has our branch given you a quotation?",
    "step3_hi":   "आप कितने BHK शिफ्ट कर रहे हैं?",
    "step3_en":   "How many BHK are you shifting?",
    "step4_hi":   "पिकअप कौन से फ्लोर पर है और लिफ्ट है?",
    "step4_en":   "Which floor is the pickup, and is there a lift?",
    "step5_hi":   "कोटेशन WhatsApp पर भेजूं या ईमेल पर?",
    "step5_en":   "Should I send the quotation on WhatsApp or email?",
    "step6_hi":   "पिकअप का पूरा पता और पिनकोड बताएं।",
    "step6_en":   "Please share the full pickup address with pincode.",
    "step7_hi":   "सर्वे के लिए कौन सा दिन ठीक रहेगा?",
    "step7_en":   "Which day works for the survey?",
    "close_hi":   "धन्यवाद, सर्वे शेड्यूल हो गया है।",
    "close_en":   "Thank you, the survey has been scheduled.",
    "unclear_hi": "कृपया दोबारा बोलें।",
    "unclear_en": "Could you please repeat that?",
}

# text.strip() → mp3 filename
_audio_cache: dict[str, str] = {}

OUTBOUND_OPENING = (
    "नमस्ते। मैं सोनी बोल रही हूं, शर्मा लॉजिस्टिक्स की तरफ से। "
    "आपकी घर शिफ्टिंग इन्क्वायरी के संबंध में कॉल कर रही हूं। "
    "क्या अभी आप एक मिनट बात कर सकते हैं?"
)
_opening_audio: str | None = None

SYSTEM_PROMPT = """You are SONY, a voice assistant for Sharma Logistics (phone call).

LANGUAGE RULES (STRICT):
- Default language: Hindi
- If user speaks English → reply in English only
- If user speaks Marathi → reply in Marathi only
- NEVER mix languages in one reply.
- One language per reply, always.

LENGTH RULE (CRITICAL):
- Maximum 1 sentence per reply. Hard limit.
- Short sentences only. This is a phone call.

CONVERSATION FLOW (ask questions one at a time, in order):

STEP 1 — Language check (only if user hasn't established language):
  Hindi: "आप किस भाषा में बात करना चाहेंगे — हिंदी, इंग्लिश या मराठी?"
  English: "Which language would you prefer — Hindi, English Or Marathi?"
  Marathi: "आपण कोणत्या भाषेत बोलू इच्छिता — हिंदी, इंग्लिश की मराठी?"

STEP 2 — Branch quotation:
  Hindi: "क्या हमारी ब्रांच से आपको कोटेशन मिला है?"
  English: "Has our branch given you a quotation?"
  Marathi: "आमच्या शाखेकडून तुम्हाला कोटेशन मिळाले आहे का?"

STEP 3 — BHK size:
  Hindi: "आप कितने BHK शिफ्ट कर रहे हैं?"
  English: "How many BHK are you shifting?"
  Marathi: "आपण किती BHK शिफ्ट करत आहात?"

STEP 4 — Floor and lift:
  Hindi: "पिकअप कौन से फ्लोर पर है और लिफ्ट है?"
  English: "Which floor is the pickup, and is there a lift?"
  Marathi: "पिकअप कोणत्या मजल्यावर आहे आणि लिफ्ट आहे का?"

STEP 5 — Quotation channel:
  Hindi: "कोटेशन WhatsApp पर भेजूं या ईमेल पर?"
  English: "Should I send the quotation on WhatsApp or email?"
  Marathi: "माझ्या व्हाट्सअपवर कोटेशन पाठवायचे का आणि ईमेल पर?"

STEP 6 — Address:
  Hindi: "पिकअप का पूरा पता और पिनकोड बताएं।"
  English: "Please share the full pickup address with pincode."
  Marathi: "कृपया पिकअपचा पूर्ण पत्ता आणि पिनकोड सांगा."

STEP 7 — Survey time:
  Hindi: "सर्वे के लिए कौन सा दिन ठीक रहेगा?"
  English: "Which day works for the survey?"
  Marathi: "सर्वेसाठी कोणता दिवस योग्य राहील?"

CLOSING:
  Hindi: "धन्यवाद, सर्वे शेड्यूल हो गया है।"
  English: "Thank you, the survey has been scheduled."
  Marathi: "धन्यवाद, सर्वे शेड्यूल करण्यात आला आहे."

IMPORTANT: Use EXACTLY the phrases above word-for-word. Do not paraphrase them.
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
    """Return cached mp3 filename if it still exists on disk."""
    filename = _audio_cache.get(text.strip())
    if filename and os.path.isfile(os.path.join(AUDIO_DIR, filename)):
        return filename
    return None


def _cache_consume(text: str) -> str | None:
    """Get cached file and schedule background re-render."""
    filename = _cache_get(text)
    if filename:
        _audio_cache.pop(text.strip(), None)  # consume
        asyncio.create_task(_rerender_bg(text))  # refill
    return filename


async def _rerender_bg(text: str):
    """Re-render a used cache entry so next call finds it ready."""
    try:
        filename = await generate_tts_async(text)
        _audio_cache[text.strip()] = filename
        log.debug("Cache refilled: %s", text[:40])
    except Exception as e:
        log.warning("Cache re-render failed: %s", e)


async def _prerender_all():
    """Render all fixed responses concurrently at startup."""
    async def _one(key: str, text: str):
        try:
            fn = await generate_tts_async(text)
            _audio_cache[text.strip()] = fn
            log.info("  ✅ [%s] cached", key)
        except Exception as e:
            log.warning("  ❌ [%s] failed: %s", key, e)

    await asyncio.gather(*[_one(k, v) for k, v in FIXED_RESPONSES.items()])


async def _run_pipeline(messages: list[dict], call_sid: str, user_input: str) -> tuple[str, str]:
    """LLM → cache check → TTS. Returns (filename, reply_text)."""
    t0 = time.perf_counter()

    # Fully buffer LLM response (55 tokens = ~20-40ms extra vs streaming,
    # but lets us check cache before calling TTS — worth it on cache hit)
    stream = await groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=55,
        temperature=0.1,   # low = deterministic = higher cache hit rate
        stream=True,
    )
    log.info("⏱  Groq open: %.3fs", time.perf_counter() - t0)

    ai_reply = ""
    async for chunk in stream:
        ai_reply += chunk.choices[0].delta.content or ""
    ai_reply = ai_reply.strip() or "कृपया दोबारा बोलें।"
    log.info("⏱  LLM done: %.3fs | %s", time.perf_counter() - t0, ai_reply[:60])

    # Cache hit?
    filename = _cache_consume(ai_reply)
    if filename:
        log.info("⚡ CACHE HIT — 0ms TTS (total: %.3fs)", time.perf_counter() - t0)
    else:
        log.info("Cache miss — calling TTS REST")
        filename = await generate_tts_async(ai_reply)
        log.info("⏱  TTS done: %.3fs", time.perf_counter() - t0)

    if call_sid:
        hist = _call_history.setdefault(call_sid, [])
        hist.append({"role": "user",      "content": user_input})
        hist.append({"role": "assistant", "content": ai_reply})

    log.info("⏱  pipeline total: %.3fs", time.perf_counter() - t0)
    return filename, ai_reply


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
        history  = _call_history.get(call_sid, [])
        is_cont  = len(history) > 0

        if not speech or not speech.strip():
            if is_cont:
                user_input = "[unclear]"
            elif _is_inbound(direction):
                user_input = "[inbound_start]"
            else:
                user_input = "[outbound_start]"
        else:
            user_input = speech.strip()

        # Fast path: serve pre-rendered "please repeat" for unclear speech
        audio_filename = None
        ai_reply       = None
        if user_input == "[unclear]":
            for key in ("unclear_hi", "unclear_en"):
                text = FIXED_RESPONSES[key]
                fn   = _cache_consume(text)
                if fn:
                    audio_filename = fn
                    ai_reply       = text
                    log.info("⚡ Unclear fast path from cache")
                    break

        if not audio_filename:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for msg in history[-10:]:
                messages.append(msg)
            messages.append({"role": "user", "content": user_input})
            audio_filename, ai_reply = await _run_pipeline(messages, call_sid, user_input)

        audio_url     = f"{ngrok_url}/audio/{audio_filename}"
        record_action = f"{ngrok_url}/voice"

        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Play>{audio_url}</Play>
    <Gather input="speech" action="{record_action}" method="POST"
            speechTimeout="1" timeout="5" actionOnEmptyResult="true"
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
            speechTimeout="1" timeout="5" actionOnEmptyResult="true" language="hi-IN" />
</Response>"""
        return Response(content=twiml, media_type="text/xml")


# ── Lifecycle ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global _opening_audio
    log.info("Starting up — warming WS + pre-rendering responses…")

    await warmup_ws()

    # Run opening + all fixed responses in parallel
    opening_task   = asyncio.create_task(generate_tts_async(OUTBOUND_OPENING))
    prerender_task = asyncio.create_task(_prerender_all())

    _opening_audio = await opening_task
    log.info("✅ Opening audio ready: %s", _opening_audio)

    await prerender_task
    log.info("✅ Startup done. Cache: %d/%d entries", len(_audio_cache), len(FIXED_RESPONSES))


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
    if not filename.endswith(".mp3") or ".." in filename or "/" in filename:
        return Response(status_code=404)
    filepath = os.path.join(AUDIO_DIR, filename)
    if not os.path.isfile(filepath):
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

    if _opening_audio and os.path.isfile(os.path.join(AUDIO_DIR, _opening_audio)):
        audio_filename = _opening_audio
        _opening_audio = None
    else:
        try:
            audio_filename = await generate_tts_async(OUTBOUND_OPENING)
        except Exception as e:
            return JSONResponse({"error": f"TTS failed: {e}"}, status_code=500)

    audio_url     = f"{ngrok_url}/audio/{audio_filename}"
    record_action = f"{ngrok_url}/voice"
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"<Play>{audio_url}</Play>"
        f'<Gather input="speech" action="{record_action}" method="POST" '
        'speechTimeout="1" timeout="5" actionOnEmptyResult="true" '
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