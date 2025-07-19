from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import logging
import random
from datetime import datetime, timedelta
from collections import defaultdict
from threading import Lock
from typing import Dict, List, Tuple

app = Flask(__name__)
CORS(app)

# Logging setup
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO
)

# Load model and tokenizer once globally
tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

# Crisis keywords for immediate flagging
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end it", "hurt myself", "cut myself",
    "want to die", "can't go on", "give up", "life is pointless", "no reason to live"
]

CONFIDENCE_THRESHOLD = 0.4

# Simple token bucket rate limit per IP (max 1 request every 2 seconds)
RATE_LIMIT_SECONDS = 2
user_last_request: Dict[str, datetime] = defaultdict(lambda: datetime.min)
rate_limit_lock = Lock()

# Resource URLs - replace these URLs with trusted, verified mental health resources
resources: Dict[str, str] = {
    "sadness": "https://www.mentalhealth.gov/talk/feelings/sadness",
    "joy": "https://www.helpguide.org/articles/mental-health/positive-psychology.htm",
    "anger": "https://www.adaa.org/understanding-anxiety/related-illnesses/anger-management",
    "fear": "https://www.nimh.nih.gov/health/topics/anxiety-disorders",
    "surprise": "https://psychcentral.com/health/what-is-surprise",
    "disgust": "https://www.psychologytoday.com/us/basics/disgust",
    "love": "https://greatergood.berkeley.edu/topic/love",
    "gratitude": "https://www.health.harvard.edu/healthbeat/giving-thanks-can-make-you-happier",
    "neutral": "https://www.mentalhealth.gov/get-help/immediate-help",
    "relief": "https://psychcentral.com/lib/dealing-with-relief-after-stress",
    "curiosity": "https://www.psychologytoday.com/us/basics/curiosity",
    "embarrassment": "https://www.psychologytoday.com/us/basics/embarrassment",
    "optimism": "https://www.helpguide.org/articles/mental-health/positive-thinking.htm",
    "crisis": "https://suicidepreventionlifeline.org/",
    "rate_limited": "https://www.mentalhealth.gov/get-help"
}

# Expanded prewritten responses per emotion
prewritten_responses: Dict[str, List[str]] = {
    "sadness": [
        "I'm really sorry you're feeling this way. Want to talk more about it?",
        "That's heavy. Remember, you're not alone, and I'm here to listen.",
        "Sadness can be overwhelming sometimes, but opening up can help.",
        "It's okay to feel sad. Would you like some suggestions to cope?",
        "Sometimes sadness comes in waves. I'm here to ride them out with you."
    ],
    "joy": [
        "That's wonderful—what's bringing you joy today?",
        "Joy is infectious—thanks for sharing your happiness!",
        "It's great to hear something positive! Care to tell me more?",
        "Moments of joy brighten our days. Glad you have one!",
        "Keep embracing these joyful moments—they're precious."
    ],
    "anger": [
        "Anger is a valid feeling. Do you want to explore what's behind it?",
        "Let's work together to understand and manage this anger.",
        "Sometimes anger signals we need a change. Want to talk about it?",
        "It's okay to feel angry—how can I support you right now?",
        "Taking deep breaths can help calm anger. Want some tips?"
    ],
    "fear": [
        "You're safe here. Can you tell me more about what’s worrying you?",
        "Fear can be intense, but talking about it helps.",
        "It’s natural to feel fear sometimes. Want to share what’s on your mind?",
        "Let's work through this fear together, step by step.",
        "Remember, fears often feel bigger in our heads than in reality."
    ],
    "surprise": [
        "Wow, that came out of nowhere! How do you feel about it?",
        "Surprises can shake us up. Want to process it together?",
        "Unexpected moments can be confusing. I'm here if you want to talk.",
        "How are you handling this surprise? Let's explore your feelings.",
        "Sometimes surprises bring growth. What do you think this means for you?"
    ],
    "disgust": [
        "That sounds unsettling—do you want to talk about it?",
        "Disgust is a strong emotion. Sharing might help.",
        "It’s okay to feel this way. What happened that brought it up?",
        "Let's explore what's causing this feeling when you're ready.",
        "Sometimes understanding disgust helps reduce its power."
    ],
    "love": [
        "That’s so heartwarming. Who or what are you referring to?",
        "Love is powerful—want to share more about it?",
        "Love brings light into our lives. Tell me more!",
        "Cherishing love is important. How does it make you feel?",
        "It’s beautiful to feel love. How does it affect you today?"
    ],
    "gratitude": [
        "Gratitude is wonderful—what are you most thankful for?",
        "Awesome! Anything else you're feeling grateful about?",
        "Gratitude can uplift our spirits. Tell me about your blessings.",
        "Recognizing good things helps us heal. What stands out for you?",
        "Keeping a gratitude journal can boost happiness. Have you tried it?"
    ],
    "neutral": [
        "Thanks for sharing. How has your day been so far?",
        "I'm here for anything on your mind—big or small.",
        "Sometimes just talking helps. What's on your mind?",
        "Feel free to share anything you're thinking or feeling.",
        "I’m listening whenever you want to chat."
    ],
    "relief": [
        "Great to hear you’re feeling relieved! What helped?",
        "That sounds like a weight off your shoulders.",
        "Relief can be refreshing. What changed for you?",
        "It’s important to acknowledge relief—what made the difference?",
        "Glad you’re feeling better. Let’s keep that going!"
    ],
    "curiosity": [
        "Tell me more—curiosity is such a great spark!",
        "What are you curious about? Let's explore together.",
        "Curiosity opens doors. What questions do you have?",
        "Exploring new things can be exciting. Want to share?",
        "Curious minds grow strong. What caught your attention?"
    ],
    "embarrassment": [
        "That sounds awkward—want to talk it through?",
        "We’ve all had those moments. You're not alone.",
        "Embarrassment can sting, but it usually fades quickly.",
        "Want to share what happened? Sometimes it helps to laugh it off.",
        "It’s okay to feel embarrassed—how can I support you?"
    ],
    "optimism": [
        "Such positivity! What’s got you feeling hopeful?",
        "Optimism is powerful—what are you most excited about?",
        "Looking on the bright side can help us cope. Tell me more.",
        "Hope fuels us forward. What are you hoping for?",
        "Staying optimistic can make a big difference. Keep it up!"
    ],
    "rate_limited": [
        "You're sending messages too quickly. Let's slow down a bit.",
        "Thanks for your patience—I'm here when you're ready.",
        "Taking a moment to breathe can help us chat better.",
        "Let's give ourselves some space between messages.",
        "I appreciate your understanding. Ready when you are!"
    ],
    "crisis": [
        "I’m really sorry. You're not alone. Can I help connect you to support?",
        "If you're thinking of harming yourself, please reach out for help now.",
        "Your feelings are important. Let's find the help you deserve.",
        "Remember, there are people ready to listen and help 24/7.",
        "You matter. Please consider talking to a professional or hotline."
    ]
}

def is_rate_limited(ip: str) -> bool:
    with rate_limit_lock:
        now = datetime.utcnow()
        if now - user_last_request[ip] < timedelta(seconds=RATE_LIMIT_SECONDS):
            return True
        user_last_request[ip] = now
    return False

def detect_crisis(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in CRISIS_KEYWORDS)

def predict_emotion(text: str) -> Tuple[str, float]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        confidence, label_id = torch.max(probs, dim=1)
        emotion = model.config.id2label[label_id.item()].lower()
    return emotion, confidence.item()

def log_user_message(ip: str, message: str, emotion: str, confidence: float):
    logging.info(f"User IP: {ip} | Message: {message} | Emotion: {emotion} | Confidence: {confidence:.2f}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    user_ip = request.remote_addr or "unknown"

    if not user_input:
        return jsonify({
            "response": "It seems empty—can you share how you're feeling?",
            "emotion": "none",
            "resource": resources.get("neutral")
        })

    if is_rate_limited(user_ip):
        logging.warning(f"Rate limit triggered for IP {user_ip}")
        return jsonify({
            "response": random.choice(prewritten_responses["rate_limited"]),
            "emotion": "rate_limited",
            "resource": resources.get("rate_limited")
        })

    if detect_crisis(user_input):
        logging.warning(f"Crisis detected from IP {user_ip}: {user_input}")
        return jsonify({
            "response": random.choice(prewritten_responses["crisis"]),
            "emotion": "crisis",
            "resource": resources.get("crisis")
        })

    try:
        emotion, confidence = predict_emotion(user_input)
        logging.info(f"IP {user_ip} | Emotion: {emotion} | Confidence: {confidence:.2f}")

        if confidence < CONFIDENCE_THRESHOLD:
            emotion = "neutral"
            reply = "I'm not quite sure how you're feeling, but I'm here to listen."
        else:
            responses = prewritten_responses.get(emotion, prewritten_responses["neutral"])
            reply = random.choice(responses)

        resource = resources.get(emotion, resources["neutral"])
        log_user_message(user_ip, user_input, emotion, confidence)

        return jsonify({
            "response": reply,
            "emotion": emotion,
            "confidence": f"{confidence:.2f}",
            "resource": resource,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })

    except Exception as e:
        logging.error(f"Emotion detection error from IP {user_ip}: {e}", exc_info=True)
        return jsonify({
            "response": "Something went wrong while analyzing your message. But I'm here to support you.",
            "emotion": "unknown",
            "resource": resources.get("neutral")
        })

if __name__ == "__main__":
    app.run(debug=True)