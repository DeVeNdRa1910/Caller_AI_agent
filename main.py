import logging
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env before any module that reads os.getenv() at import time

from fastapi import FastAPI, Request
from fastapi.responses import Response, FileResponse, JSONResponse
from groq import Groq
from twilio.rest import Client
from tts import generate_tts, AUDIO_DIR

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Per-call conversation history: CallSid -> list of {"role": "user"|"assistant", "content": "..."}
# So the agent continues the conversation instead of repeating the intro when user replies.
_call_history: dict[str, list[dict[str, str]]] = {}

SYSTEM_PROMPT = """
You are SONY, a polite, professional AI voice assistant representing Sharma Logistics. Your goal is to qualify a household shifting enquiry, collect required details, build trust, and schedule a free home survey. LANGUAGE INSTRUCTION: You must be able to speak in both Hindi and English. At the beginning of the call, always ask the user for their preferred language. Continue the entire conversation in the selected language. If the user switches language during the conversation, adapt naturally and continue in that language. If the user is comfortable with both, use simple Hindi mixed with easy English (Hinglish). Always maintain a natural, patient, and respectful tone in either language. Speak clearly, naturally, patiently, and respectfully. Maintain a warm, helpful, and professional tone at all times. FLOW: Opening + Language Selection: नमस्ते। मैं सोनी बोल रही हूं, शर्मा लॉजिस्टिक्स की तरफ से। मैं आपके घर शिफ्टिंग की इन्क्वायरी में मदद करने वाली AI सहायक हूं। क्या अभी आप एक मिनट बात कर सकते हैं? If the user agrees: धन्यवाद। आगे बढ़ने से पहले, आप किस भाषा में बात करना पसंद करेंगे — हिंदी या इंग्लिश? If user chooses Hindi → continue the entire conversation in Hindi. If user chooses English → continue the entire conversation in English. If user says both → continue in simple Hindi mixed with easy English. If user changes language later → adapt automatically and continue in that language. If the user says it is NOT a good time: कोई बात नहीं। जब भी सुविधा हो कृपया कॉल कर लें। धन्यवाद। End the call politely. Purpose of the Call: (If Hindi selected) धन्यवाद। मैं आपकी इन्दौर, मध्य प्रदेश से पुणे, महाराष्ट्र तक घर का सामान शिफ्ट करने की इन्क्वायरी के बारे में कॉल कर रही हूं। बस कुछ बातें पक्की कर लूं ताकि हम ठीक से मदद कर सकें। (If English selected) Thank you. I am calling regarding your enquiry for shifting your household items from Indore, Madhya Pradesh to Pune, Maharashtra. I just need to confirm a few details so that we can assist you properly. Q1 – Branch Contact Status: (Hindi) क्या हमारी ब्रांच से किसी ने आपको पहले ही कॉल किया है, आर्टिकल लिस्ट ली है और कोटेशन भेजा है? (English) Has anyone from our branch already called you, taken the article list, and shared a quotation? If the answer is NO: (Hindi) आपसे देरी के लिए सच में माफी चाहती हूं। हम तुरंत आपकी मदद करेंगे। (English) I sincerely apologize for the delay. We will assist you immediately. Q2 – Household Size: (Hindi) आप एक BHK, दो BHK या तीन BHK शिफ्ट कर रहे हैं? (English) Are you shifting a 1 BHK, 2 BHK, or 3 BHK household? After response: (Hindi) धन्यवाद, पक्का करने के लिए। (English) Thank you for confirming. Q3 – Move Details: (Hindi) सही अनुमान देने के लिए कृपया बताएं: पिकअप का फ्लोर नंबर क्या है? लिफ्ट है या नहीं? और कौन-सी गाड़ियां शिफ्ट करनी हैं – जैसे कार या बाइक? (English) To give you an accurate estimate, please tell me: What is the pickup floor number? Is there a lift available? And are any vehicles being shifted, such as a car or bike? After response: (Hindi) इन जानकारियों के लिए धन्यवाद। (English) Thank you for the information. Q4 – Quotation Preference: (Hindi) आप कोटेशन ईमेल पर चाहेंगे या व्हाट्सऐप पर? (English) Would you like to receive the quotation on email or WhatsApp? If Email: (Hindi) कृपया अपना ईमेल पता बताएं। (English) Please share your email address. If WhatsApp: (Hindi) बढ़िया। क्या यह आपका व्हाट्सऐप नंबर है जो 45 पर खत्म होता है? मैंने आपको मैसेज भेज दिया है – जब सुविधा हो वहां अपनी आर्टिकल लिस्ट भेज दें ताकि हम अनुमान बना सकें। (English) Great. Is this your WhatsApp number ending with 45? I have sent you a message. Please share your article list there whenever convenient so we can prepare an estimate. Q5 – Address Collection: (Hindi) कृपया अपना पूरा पिकअप पता पिनकोड के साथ बताएं ताकि मैं आपके लिए मुफ्त होम सर्वे की व्यवस्था कर सकूं। (English) Please share your complete pickup address along with the pincode so that I can arrange a free home survey for you. After response: (Hindi) पता शेयर करने के लिए धन्यवाद। (English) Thank you for sharing the address. Q6 – Survey Scheduling: (Hindi) किस दिन और समय पर आपके लिए सुविधाजनक रहेगा कि हमारा एक्जीक्यूटिव आकर सामान देख सके? (English) Which day and time would be convenient for you for our executive to visit and inspect the items? Confirmation: (Hindi) ठीक है। मैंने आपका मुफ्त होम सर्वे कल शाम पाँच बजे के बाद शेड्यूल कर दिया है। (English) Alright. I have scheduled your free home survey for tomorrow after 5 PM. Trust-Building Statement: (Hindi) बस आपको बता दूं, सर्वे के दौरान हमारा फील्ड ऑफिसर सामान की सुरक्षित पैकिंग के लिए सब चेक करेगा, इंश्योरेंस के विकल्प बताएगा, कोई छुपा चार्ज नहीं – पूरी पारदर्शी कोटेशन देगा, और डिलीवरी व सुरक्षा से जुड़े सभी सवालों का जवाब देगा। (English) Just to inform you, during the survey our field officer will check all items for safe packing, explain insurance options, provide a fully transparent quotation with no hidden charges, and answer all your questions related to delivery and safety. Objection Handling (If user says rates might be high): (Hindi) आपकी बात समझ आती है। इसीलिए शर्मा लॉजिस्टिक्स मुफ्त निरीक्षण देता है। आप सिर्फ उतने सामान का भुगतान करेंगे जितना आप शिफ्ट करवाते हैं – कोई अतिरिक्त चार्ज नहीं। अंतिम कोटेशन सामान और दूरी के हिसाब से होगा। (English) I understand your concern. That is why Sharma Logistics provides a free inspection. You only pay for the items you actually move — there are no extra charges. The final quotation depends on the items and distance. Wrap-Up: (Hindi) आपका समय देने के लिए बहुत धन्यवाद। मैंने सारी जानकारी नोट कर ली है और आपका सर्वे शेड्यूल कर दिया है। हमारा इन्दौर ब्रांच एक्जीक्यूटिव कल शाम पाँच बजे के बाद आपके यहां निरीक्षण करने और अंतिम कोटेशन देने आएगा। उससे पहले किसी भी सहायता के लिए आप व्हाट्सऐप पर जवाब दे सकते हैं। आपका दिन शुभ रहे। (English) Thank you very much for your time. I have noted all the details and scheduled your survey. Our Indore branch executive will visit tomorrow after 5 PM for inspection and final quotation. You can reply on WhatsApp if you need any assistance before that. Have a great day.
"""

def _is_inbound(direction: str | None) -> bool:
    """True if the user called us (inbound); False if we called the user (outbound)."""
    if not direction:
        return True  # default to inbound greeting
    return direction.strip().lower() == "inbound"


def _escape_say(text: str) -> str:
    """Escape text for safe use inside Twilio <Say> (XML)."""
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _error_twiml(message: str) -> str:
    """TwiML for exception path only: short Say so user hears something. Normal flow uses Sarvam <Play> only."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="en-IN">{_escape_say(message)}</Say>
    <Gather input="speech" action="{os.getenv('NGROK_URL', '').rstrip('/')}/voice/start" method="POST" speechTimeout="auto" timeout="3" actionOnEmptyResult="true" language="en-IN" />
</Response>"""


def _form_get(form_like, key: str, default=None):
    """Get form value; try exact key then lowercase (Twilio sends capitalized)."""
    v = form_like.get(key)
    if v is not None and str(v).strip() != "":
        return v
    if key != key.lower():
        v = form_like.get(key.lower())
        if v is not None and str(v).strip() != "":
            return v
    return default


async def _handle_voice(form_like) -> Response:
    """Shared logic for /voice: form_like must have .get(k, default)."""
    ngrok_url = os.getenv("NGROK_URL", "https://your-ngrok-url").rstrip("/")
    direction = _form_get(form_like, "Direction") or ""
    call_sid = (_form_get(form_like, "CallSid") or "").strip()
    user_input = _form_get(form_like, "SpeechResult")
    confidence = _form_get(form_like, "Confidence")

    log.info(
        "Voice webhook: direction=%s CallSid=%s speech_len=%s confidence=%s",
        direction, call_sid, len((user_input or "").strip()), confidence,
    )

    try:
        history = _call_history.get(call_sid, [])
        is_continuation = len(history) > 0
        if user_input and user_input.strip():
            log.info("Twilio STT: %s (confidence=%s)", user_input.strip()[:160], confidence)

        if not user_input or not user_input.strip():
            if is_continuation:
                user_input = "[Context: The user just spoke but we could not hear them clearly (or the speech result was empty). Politely ask them to repeat what they said, in one short sentence.]"
            else:
                if _is_inbound(direction):
                    user_input = "[Context: The user has just called our company number. Greet them and ask how you can help.]"
                else:
                    user_input = "[Context: You have just called the user. Introduce yourself as SONY from Sharma Logistics and say you are calling regarding their household shifting enquiry. Ask if they have a moment to talk.]"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Keep last 10 turns (20 messages) to avoid token limits
        for msg in history[-20:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_input})

        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
        )
        ai_reply = (completion.choices[0].message.content or "").strip()
        if not ai_reply:
            ai_reply = "I did not get a response. Please try again."
        log.info("LLM reply length=%s", len(ai_reply))

        # Append to per-call history so next user reply continues the conversation
        if call_sid:
            if call_sid not in _call_history:
                _call_history[call_sid] = []
            _call_history[call_sid].append({"role": "user", "content": user_input})
            _call_history[call_sid].append({"role": "assistant", "content": ai_reply})

        # Sarvam TTS only — no Twilio <Say>. LLM controls content; Sarvam speaks it.
        audio_filename = generate_tts(ai_reply)
        audio_url = f"{ngrok_url}/audio/{audio_filename}"
        record_action = f"{ngrok_url}/voice/start"
        # Twilio speech gather (no recording fetch / external STT needed).
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Play>{audio_url}</Play>
    <Gather input="speech" action="{record_action}" method="POST" speechTimeout="auto" timeout="3" actionOnEmptyResult="true" language="en-IN" />
</Response>"""
        return Response(content=twiml, media_type="text/xml")
    except Exception as e:
        log.exception("Voice handler error: %s", e)
        fallback = _error_twiml("We are having a technical issue. Please try again in a moment.")
        return Response(content=fallback, media_type="text/xml")


def _redirect_twiml(start_url: str) -> str:
    """Instant response: redirect to /voice/start so Twilio never times out. No Say."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Redirect method="POST">{start_url}</Redirect>
</Response>"""


@app.post("/voice")
async def voice_webhook(request: Request):
    """
    First webhook when call connects. Returns immediately (no LLM) then redirects to /voice/start.
    This avoids Twilio timeout while we run LLM. Set this URL in Twilio as the "A call comes in" webhook.
    """
    ngrok_url = os.getenv("NGROK_URL", "https://your-ngrok-url").rstrip("/")
    return Response(
        content=_redirect_twiml(f"{ngrok_url}/voice/start"),
        media_type="text/xml",
    )


@app.post("/voice/start")
async def voice_start_post(request: Request):
    """
    Runs after redirect from /voice and after each user speech gather.
    Does LLM → Sarvam TTS → Play + Gather. Twilio POSTs here (redirect and Gather action).
    """
    form = await request.form()
    return await _handle_voice(form)


@app.post("/voice/fallback")
async def voice_fallback():
    """Optional: set as Fallback URL in Twilio so user hears a message if the main voice URL fails."""
    log.warning("Voice fallback hit - main URL may have failed")
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="en-IN">We could not connect the assistant. Please try again later. Goodbye.</Say>
    <Hangup/>
</Response>"""
    return Response(content=twiml, media_type="text/xml")


@app.post("/voice/status")
async def voice_status(request: Request):
    """Twilio status callback (answered, completed). Log and clear conversation when call ends."""
    try:
        body = await request.form() if request.method == "POST" else request.query_params
        body_dict = dict(body)
        log.info("Call status: %s", body_dict)
        call_sid = body_dict.get("CallSid")
        status = (body_dict.get("CallStatus") or "").strip().lower()
        if call_sid and status == "completed":
            _call_history.pop(call_sid, None)
    except Exception:
        pass
    return Response(content="", status_code=200)


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve generated TTS MP3 for Twilio Play. Files live in audio_files/."""
    if not filename.endswith(".mp3") or ".." in filename or "/" in filename:
        return Response(status_code=404)
    filepath = os.path.join(AUDIO_DIR, filename)
    if not os.path.isfile(filepath):
        log.warning("Audio file not found: %s", filepath)
        return Response(status_code=404)
    return FileResponse(filepath, media_type="audio/mpeg")


def _get_first_outbound_message() -> str:
    """Get the LLM's first message for outbound call (used in inline TwiML)."""
    user_ctx = "[Context: You have just called the user. Introduce yourself as SONY from Sharma Logistics and say you are calling regarding their household shifting enquiry. Ask if they have a moment to talk. Keep your reply brief and under 300 words.]"
    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_ctx},
        ],
    )
    reply = (completion.choices[0].message.content or "").strip()
    return reply[:3000] if len(reply) > 3000 else reply  # TwiML limit 4000 chars


@app.post("/call-user")
def call_user(mobile_number: str):
    """
    Outbound: AI agent calls the user.
    First message: LLM → Sarvam TTS → inline TwiML with <Play> (no Polly).
    When user answers, Twilio plays Sarvam MP3 then gathers speech (Twilio STT).
    """
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_number = os.getenv("TWILIO_PHONE_NUMBER")
    ngrok_url = os.getenv("NGROK_URL", "").rstrip("/")

    if not ngrok_url:
        return JSONResponse(content={"error": "NGROK_URL not set in .env"}, status_code=400)

    log.info("Generating first message and Sarvam TTS for outbound call...")
    try:
        first_message = _get_first_outbound_message()
        audio_filename = generate_tts(first_message)
    except Exception as e:
        log.exception("LLM or Sarvam TTS failed for first message: %s", e)
        first_message = "Hello, this is SONY from Sharma Logistics. We are calling about your shifting enquiry. Do you have a moment to talk?"
        try:
            audio_filename = generate_tts(first_message)
        except Exception as tts_err:
            log.exception("Fallback TTS also failed: %s", tts_err)
            return JSONResponse(
                content={"error": "TTS failed. Check SARVAM_API_KEY and Sarvam API (403 = invalid key or quota)."},
                status_code=500,
            )

    audio_url = f"{ngrok_url}/audio/{audio_filename}"
    record_action = f"{ngrok_url}/voice/start"
    # Inline TwiML: Sarvam MP3 only, no <Say>.
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?><Response><Play>{audio_url}</Play><Gather input="speech" action="{record_action}" method="POST" speechTimeout="auto" timeout="3" actionOnEmptyResult="true" language="en-IN" /></Response>"""

    status_callback = f"{ngrok_url}/voice/status" if ngrok_url else None
    twilio_client = Client(account_sid, auth_token)
    call = twilio_client.calls.create(
        to=mobile_number,
        from_=twilio_number,
        twiml=twiml,
        status_callback=status_callback,
        status_callback_event=["completed"],
    )

    log.info("Outbound call created: %s", call.sid)
    return {"status": "calling", "call_sid": call.sid}
