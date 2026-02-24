import logging
import os
import asyncio
import uuid
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse
from groq import AsyncGroq
from twilio.rest import Client
from tts import generate_tts_async, close_http_client, AUDIO_DIR

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI()

groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

_call_history: dict[str, list[dict[str, str]]] = {}

SYSTEM_PROMPT = """

CRITICAL LANGUAGE POLICY (MANDATORY – OVERRIDES ALL OTHER INSTRUCTIONS):

1. Always reply in the same language as the user.
2. Hindi → reply in Hindi
3. English → reply in English
4. Marathi → reply in Marathi
5. Mixed Hindi+English → reply in simple Hinglish
6. If unsure, ask:"आप किस भाषा में बात करना चाहेंगे — हिंदी, इंग्लिश या मराठी?"
7. Continue in the chosen language unless user switches.
8. Keep responses VERY short: 1–2 sentences max. Phone-friendly only.

------------------------------------------------------------

You are SONY, a polite and professional AI voice assistant representing Sharma Logistics.

Goal:
- Qualify household shifting enquiry
- Collect details
- Build trust
- Schedule free home survey

Always speak clearly, patiently, respectfully and conversationally.
CRITICAL: Keep every reply to 1-2 short sentences only. Never list multiple questions at once.

------------------------------------------------------------
OPENING (Inbound):नमस्ते। मैं सोनी बोल रही हूं, शर्मा लॉजिस्टिक्स की तरफ से। क्या अभी आप एक मिनट बात कर सकते हैं?
OPENING (Outbound):नमस्ते। मैं सोनी बोल रही हूं, शर्मा लॉजिस्टिक्स की तरफ से। आपकी घर शिफ्टिंग इन्क्वायरी के संबंध में कॉल कर रही हूं। क्या अभी आप एक मिनट बात कर सकते हैं?
If not good time:कोई बात नहीं। जब भी सुविधा हो कृपया कॉल कर लें। धन्यवाद।
If agrees:आगे बढ़ने से पहले, आप किस भाषा में बात करना पसंद करेंगे — हिंदी, इंग्लिश या मराठी?

------------------------------------------------------------
PURPOSE OF CALL

Hindi:धन्यवाद। मैं आपकी इंदौर से पुणे घर शिफ्टिंग इन्क्वायरी के बारे में कॉल कर रही हूं। कुछ विवरण पक्के करने हैं।
English:Thank you. I am calling regarding your enquiry for shifting your household items from Indore to Pune. I need to confirm a few details.
Marathi:धन्यवाद. मी इंदौर ते पुणे घरगुती सामान शिफ्ट करण्याच्या तुमच्या चौकशीबद्दल कॉल करत आहे. काही तपशील पुष्टी करायचे आहेत.

------------------------------------------------------------
QUESTIONS FLOW (one by one, strictly one question per turn)

Q1 – Branch Contact  
Hindi: क्या हमारी ब्रांच से किसी ने कॉल करके कोटेशन दिया है?  
English: Has anyone from our branch shared a quotation?  
Marathi: आमच्या ब्रांचमधून कुणी तुम्हाला कोटेशन दिले आहे का?

If NO:
Hindi: देरी के लिए माफी चाहती हूं, हम तुरंत मदद करेंगे।  
English: Apologies for the delay, we will assist immediately.  
Marathi: उशीराबद्दल माफी असावी, आम्ही लगेच मदत करू.

------------------------------------------------------------
Q2 – Household Size  
Hindi: आप 1 BHK, 2 BHK या 3 BHK शिफ्ट कर रहे हैं?  
English: Are you shifting 1, 2 or 3 BHK?  
Marathi: तुम्ही 1, 2 की 3 BHK शिफ्ट करत आहात?

------------------------------------------------------------
Q3 – Move Details  
Hindi: पिकअप फ्लोर? लिफ्ट? कोई वाहन?  
English: Pickup floor? Lift? Any vehicle?  
Marathi: पिकअप फ्लोअर? लिफ्ट आहे का? वाहन आहे का?

------------------------------------------------------------
Q4 – Quotation Preference  
Hindi: कोटेशन ईमेल या व्हाट्सऐप?  
English: Email or WhatsApp quotation?  
Marathi: कोटेशन ईमेलवर की WhatsApp वर?

------------------------------------------------------------
Q5 – Address  
Hindi: पूरा पिकअप पता पिनकोड सहित बताएं।  
English: Share pickup address with pincode.  
Marathi: कृपया पिकअप पत्ता पिनकोडसह सांगा.

------------------------------------------------------------
Q6 – Survey Schedule  
Hindi: सर्वे के लिए कौन सा दिन और समय ठीक रहेगा?  
English: Convenient day and time for survey?  
Marathi: सर्वेसाठी कोणता दिवस आणि वेळ सोयीचा आहे?

------------------------------------------------------------
TRUST STATEMENT
Hindi:सर्वे में सुरक्षित पैकिंग, इंश्योरेंस और पारदर्शी कोटेशन बताया जाएगा। कोई छुपा चार्ज नहीं।
English:Survey includes safe packing, insurance and transparent quotation with no hidden charges.
Marathi:सर्वेमध्ये सुरक्षित पॅकिंग, विमा आणि पारदर्शक कोटेशन दिले जाईल. कोणतेही लपलेले शुल्क नाही.

------------------------------------------------------------
PRICE CONCERN
Hindi:आप सिर्फ उतने सामान का भुगतान करेंगे जितना शिफ्ट होगा।
English:You only pay for items you move.
Marathi:तुम्ही फक्त शिफ्ट होणाऱ्या सामानाचेच पैसे द्याल.

------------------------------------------------------------
CLOSING
Hindi:धन्यवाद। आपकी जानकारी नोट कर ली है और सर्वे शेड्यूल कर दिया है।
English:Thank you. I have noted your details and scheduled the survey.
Marathi:धन्यवाद. तुमची माहिती नोंदवली असून सर्वे शेड्यूल केला आहे.
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
    LLM → TTS pipeline, fully async.
    Both calls use persistent connections (AsyncGroq + httpx keep-alive).
    Returns the audio filename.
    """
    # LLM — capped at 80 tokens (1–2 sentences max → shorter = faster TTS)
    t0 = asyncio.get_event_loop().time()
    completion = await groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=80,
        temperature=0.3,
    )
    ai_reply = (completion.choices[0].message.content or "").strip()
    if not ai_reply:
        ai_reply = "कृपया दोबारा बोलें।"
    t1 = asyncio.get_event_loop().time()
    log.info("LLM: %.2fs → %d chars: %s", t1 - t0, len(ai_reply), ai_reply[:100])

    # Save history
    if call_sid:
        hist = _call_history.setdefault(call_sid, [])
        hist.append({"role": "user", "content": user_input})
        hist.append({"role": "assistant", "content": ai_reply})

    # TTS
    audio_filename = await generate_tts_async(ai_reply)
    t2 = asyncio.get_event_loop().time()
    log.info("TTS: %.2fs — total pipeline: %.2fs", t2 - t1, t2 - t0)

    return audio_filename


async def _handle_voice(form_like) -> Response:
    ngrok_url = os.getenv("NGROK_URL", "").rstrip("/")
    direction  = _form_get(form_like, "Direction") or ""
    call_sid   = (_form_get(form_like, "CallSid") or "").strip()
    user_input = _form_get(form_like, "SpeechResult")
    confidence = _form_get(form_like, "Confidence")

    log.info("Voice: sid=%s dir=%s conf=%s speech=%s",
             call_sid, direction, confidence,
             (user_input or "")[:80] or "<empty>")

    try:
        history = _call_history.get(call_sid, [])
        is_continuation = len(history) > 0

        if not user_input or not user_input.strip():
            if is_continuation:
                user_input = "[User spoke but audio was unclear. Ask them to repeat in one sentence.]"
            elif _is_inbound(direction):
                user_input = "[Inbound call just connected. Greet the user as SONY from Sharma Logistics.]"
            else:
                user_input = "[Outbound call just connected. Introduce yourself as SONY from Sharma Logistics and ask if they have a moment.]"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in history[-16:]:   # last 8 turns — keep context window lean
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_input})

        # ── KEY CHANGE: await directly — no polling, no redirects, no round-trips ──
        # FastAPI runs each request in the async event loop. Awaiting here is
        # non-blocking: other requests/polls are served while this one waits.
        # This eliminates ~2–4s of polling overhead from the previous approach.
        audio_filename = await _run_pipeline(messages, call_sid, user_input)

        audio_url     = f"{ngrok_url}/audio/{audio_filename}"
        record_action = f"{ngrok_url}/voice/start"

        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Play>{audio_url}</Play>
    <Gather input="speech" action="{record_action}" method="POST"
            speechTimeout="auto" timeout="3" actionOnEmptyResult="true" language="hi-IN" />
</Response>"""
        return Response(content=twiml, media_type="text/xml")

    except Exception as e:
        log.exception("Voice handler error: %s", e)
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="hi-IN">Technical issue. Please try again.</Say>
    <Gather input="speech" action="{ngrok_url}/voice/start" method="POST"
            speechTimeout="auto" timeout="3" actionOnEmptyResult="true" language="hi-IN" />
</Response>"""
        return Response(content=twiml, media_type="text/xml")


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    log.info("Server started.")


@app.on_event("shutdown")
async def shutdown():
    await close_http_client()
    log.info("HTTP client closed.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/voice")
async def voice_webhook(request: Request):
    """First webhook when call connects — redirect straight to /voice/start."""
    ngrok_url = os.getenv("NGROK_URL", "").rstrip("/")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Redirect method="POST">{ngrok_url}/voice/start</Redirect>
</Response>"""
    return Response(content=twiml, media_type="text/xml")


@app.post("/voice/start")
async def voice_start(request: Request):
    form   = await request.form()
    params = dict(request.query_params)
    merged = {**params, **dict(form)}
    return await _handle_voice(merged)


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


# ── Outbound call ─────────────────────────────────────────────────────────────

@app.post("/call-user")
async def call_user(mobile_number: str):
    account_sid   = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token    = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_number = os.getenv("TWILIO_PHONE_NUMBER")
    ngrok_url     = os.getenv("NGROK_URL", "").rstrip("/")

    if not ngrok_url:
        return JSONResponse({"error": "NGROK_URL not set"}, status_code=400)

    intro_ctx = "[Outbound call connected. Introduce yourself as SONY from Sharma Logistics and ask if they have a moment to talk. 2 sentences max.]"
    try:
        completion = await groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": intro_ctx},
            ],
            max_tokens=80,
        )
        first_msg = (completion.choices[0].message.content or "").strip()
        audio_filename = await generate_tts_async(first_msg)
    except Exception as e:
        log.exception("Outbound first message failed: %s", e)
        fallback = "Error; नमस्ते, मैं सोनी बोल रही हूं शर्मा लॉजिस्टिक्स से। क्या आप एक मिनट बात कर सकते हैं?"
        try:
            audio_filename = await generate_tts_async(fallback)
        except Exception:
            return JSONResponse({"error": "TTS failed. Check SARVAM_API_KEY."}, status_code=500)

    audio_url     = f"{ngrok_url}/audio/{audio_filename}"
    record_action = f"{ngrok_url}/voice/start"
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"<Play>{audio_url}</Play>"
        f'<Gather input="speech" action="{record_action}" method="POST" '
        'speechTimeout="auto" timeout="3" actionOnEmptyResult="true" language="hi-IN" />'
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
    log.info("Outbound call: %s", call.sid)
    return {"status": "calling", "call_sid": call.sid}