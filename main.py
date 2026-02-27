"""
main.py — Ultra-low-latency voice pipeline for Sharma Logistics SONY bot.

LATENCY STRATEGY:
  Gather path: speechTimeout 0.8s + LLM stream + TTS overlap + TTS cache.
  Media Streams path: /voice-stream → WebSocket → VAD → STT → LLM → progressive TTS (~2s to first audio).

OPTIMIZATIONS (from call-customer-support-system and local):
  1. speechTimeout="0.8" — faster end-of-speech (Gather).
  2. LLM+TTS OVERLAP: llm_to_tts_stream(stream) so first tokens go to TTS while LLM streams.
  3. TTS response cache (arbitrary text) in tts.py — repeat replies skip API.
  4. Pre-render cache for fixed phrases + unclear + opening.
  5. MEDIA STREAMS: POST /voice-stream returns <Stream>; WS /media-stream does VAD, STT,
     LLM stream, sentence-by-sentence TTS → push chunks so agent starts in ~2s.
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

# Groq model: set GROQ_MODEL in .env (free options: llama-3.1-8b-instant, openai/gpt-oss-20b, llama-3.3-70b-versatile)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Sentence boundaries only — full sentences for natural speech (no tiny fragments)
_SENTENCE_END = re.compile(r"[।.?!]\s*")
# Safety: if LLM streams this many chars without sentence end, yield anyway so first audio ~2–3s
_MAX_BUFFER_CHARS = 120

# ─────────────────────────────────────────────────────────────────────────────
# PRE-RENDERED RESPONSE CACHE
# All fixed questions + common fallbacks are rendered at startup.
# Cache entry: text.strip() → mp3 filename.
# After serving, re-renders in background so it's ready next call.
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

# text.strip() → mp3 filename
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
    """Return cached mp3 filename if it still exists on disk."""
    filename = _audio_cache.get(text.strip())
    if filename and os.path.isfile(os.path.join(AUDIO_DIR, filename)):
        return filename
    return None


def _cache_consume(text: str) -> str | None:
    """Get cached file and schedule background re-render."""
    filename = _cache_get(text)
    if filename:
        _audio_cache.pop(text.strip(), None)   # consume
        asyncio.create_task(_rerender_bg(text))  # refill immediately
    return filename


async def _rerender_bg(text: str):
    """Re-render a used cache entry so next call finds it ready."""
    try:
        filename = await synthesise_text(text)   # uses persistent WS
        _audio_cache[text.strip()] = filename
        log.debug("Cache refilled: %s", text[:40])
    except Exception as e:
        log.warning("Cache re-render failed: %s", e)


async def _prerender_all():
    """Render all fixed responses via persistent WS at startup."""
    async def _one(key: str, text: str):
        try:
            fn = await synthesise_text(text)     # uses persistent WS
            _audio_cache[text.strip()] = fn
            log.info("  ✅ [%s] cached", key)
        except Exception as e:
            log.warning("  ❌ [%s] failed: %s", key, e)

    # Serialise to avoid hammering the single persistent WS connection with
    # concurrent requests (the WS lock would queue them anyway, but this is cleaner)
    for key, text in FIXED_RESPONSES.items():
        await _one(key, text)


async def _stream_llm_sentences(messages: list[dict]):
    """Stream Groq LLM; yield only on sentence end (।.?!). Full sentences = natural speech, first audio in ~2–3s."""
    stream = await groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=128,
        temperature=0.1,
        stream=True,
        tool_choice="none",
    )
    buffer = ""
    async for chunk in stream:
        delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
        buffer += delta
        m = _SENTENCE_END.search(buffer)
        if m:
            sentence = buffer[: m.end()].strip()
            buffer = buffer[m.end() :]
            if sentence:
                yield sentence
        elif len(buffer.strip()) >= _MAX_BUFFER_CHARS:
            # Long run-on without period: yield so first audio still starts in ~2–3s
            sentence = buffer.strip()
            buffer = ""
            if sentence:
                yield sentence
    if buffer.strip():
        yield buffer.strip()


async def _run_pipeline(messages: list[dict], call_sid: str, user_input: str) -> tuple[list[str], str]:
    """Stream LLM to get sentences, then run all TTS in parallel (reference-repo style) → ~4–5s total instead of 8–9s."""
    t0 = time.perf_counter()
    # 1. Collect all sentences from stream (fast)
    sentences: list[str] = []
    async for sentence in _stream_llm_sentences(messages):
        if sentence.strip():
            sentences.append(sentence.strip())
    ai_reply = " ".join(sentences).strip() or "कृपया दोबारा बोलें।"
    if not sentences:
        stream = await groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=128,
            temperature=0.1,
            stream=True,
            tool_choice="none",
        )
        fn, ai_reply = await llm_to_tts_stream(stream)
        ai_reply = ai_reply.strip() or "कृपया दोबारा बोलें।"
        filenames = [fn]
    else:
        # 2. Parallel TTS via REST (WS has single lock; REST allows concurrent requests → ~max(TTS) not sum)
        tts_tasks = [generate_tts_async(s) for s in sentences]
        filenames = await asyncio.gather(*tts_tasks)
        filenames = list(filenames)
        log.info("⏱  parallel TTS done: %.3fs | %d segment(s)", time.perf_counter() - t0, len(filenames))
    if call_sid:
        hist = _call_history.setdefault(call_sid, [])
        hist.append({"role": "user", "content": user_input})
        hist.append({"role": "assistant", "content": ai_reply})
    log.info("⏱  pipeline total: %.3fs | %d segment(s) | %s", time.perf_counter() - t0, len(filenames), ai_reply[:60])
    return filenames, ai_reply


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
        audio_filenames: list[str] = []
        ai_reply       = None
        if user_input == "[unclear]":
            for key in ("unclear_hi", "unclear_en"):
                text = FIXED_RESPONSES[key]
                fn   = _cache_consume(text)
                if fn:
                    audio_filenames = [fn]
                    ai_reply       = text
                    log.info("⚡ Unclear fast path from cache")
                    break

        if not audio_filenames:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for msg in history[-4:]:
                messages.append(msg)
            messages.append({"role": "user", "content": user_input})
            audio_filenames, ai_reply = await _run_pipeline(messages, call_sid, user_input)

        record_action = f"{ngrok_url}/voice"
        play_elements = "".join(f'<Play>{ngrok_url}/audio/{fn}</Play>' for fn in audio_filenames)
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
    """Remove .mp3 files in audio_files older than max_age_seconds (e.g. never fetched)."""
    if not os.path.isdir(AUDIO_DIR):
        return
    now = time.time()
    removed = 0
    for name in os.listdir(AUDIO_DIR):
        if not name.endswith(".mp3"):
            continue
        path = os.path.join(AUDIO_DIR, name)
        try:
            if os.path.isfile(path) and (now - os.path.getmtime(path)) > max_age_seconds:
                os.remove(path)
                removed += 1
        except OSError as e:
            log.warning("Cleanup could not remove %s: %s", name, e)
    if removed:
        log.info("Cleaned %d stale audio file(s) from audio_files", removed)


async def _cleanup_audio_loop(interval_seconds: int = 120, max_age_seconds: int = 300):
    """Background task: periodically delete stale audio files from audio_files."""
    while True:
        await asyncio.sleep(interval_seconds)
        _cleanup_stale_audio(max_age_seconds)


@app.on_event("startup")
async def startup():
    global _opening_audio
    log.info("Starting up — establishing persistent WS + pre-rendering responses…")

    # Step 1: Connect persistent WS (blocking until done so pre-render can use it)
    await warmup_ws()

    # Step 2: Pre-render opening (block so /call-user has it). Fixed responses run in background.
    _opening_audio = await synthesise_text(OUTBOUND_OPENING)
    log.info("✅ Opening audio ready: %s", _opening_audio)

    # Step 3: Pre-render fixed responses in background so server can accept requests immediately
    async def _startup_prerender():
        await _prerender_all()
        log.info("✅ Startup cache warm. Cache: %d/%d entries", len(_audio_cache), len(FIXED_RESPONSES))

    asyncio.create_task(_startup_prerender())
    asyncio.create_task(_cleanup_audio_loop(interval_seconds=120, max_age_seconds=300))
    log.info("✅ Startup done — server ready (cache warming + audio cleanup in background).")


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
    """
    Twilio webhook that returns TwiML with <Stream> so the call uses Media Streams (WebSocket).
    Set this as the "A call comes in" webhook URL in Twilio to get ~2s "agent starts speaking"
    via progressive TTS (first sentence played while rest generates).
    """
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
    Serve TTS audio for Twilio <Play>. After the full clip has been sent (agent has
    completely delivered the audio), the file is deleted from audio_files so it
    does not accumulate.
    """
    if not filename.endswith(".mp3") or ".." in filename or "/" in filename:
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
        log.warning("Could not delete audio after serve %s: %s", filename, e)
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

    # Pre-render opening once (so Twilio can play it immediately)
    if _opening_audio and os.path.isfile(os.path.join(AUDIO_DIR, _opening_audio)):
        audio_filename = _opening_audio
        _opening_audio = None
    else:
        try:
            audio_filename = await synthesise_text(OUTBOUND_OPENING)
        except Exception as e:
            return JSONResponse({"error": f"TTS failed: {e}"}, status_code=500)

    audio_url = f"{ngrok_url}/audio/{audio_filename}"

    # Use Media Streams (WebSocket) after the opening message instead of <Gather>.
    # This matches the ultra-low-latency design from the reference repo: Twilio
    # streams audio to /media-stream, and our VAD+STT+LLM+TTS pipeline responds
    # in ~2–4s from end-of-speech.
    stream_ws = ngrok_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    if not stream_ws.endswith("/media-stream"):
        stream_ws = stream_ws.rstrip("/") + "/media-stream"

    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"<Play>{audio_url}</Play>"
        f"<Connect><Stream url=\"{stream_ws}\" /></Connect>"
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