#  Emotion Detection Chat API – Deep Dive Documentation

This project is a Flask-based RESTful API that performs **emotion detection** on user input using a fine-tuned transformer model. It classifies the emotional tone of a message and responds with **empathetic, human-like replies**, along with **resource links** for mental health support. It’s designed for ethical AI-powered support chat experiences.

---

## 📘 Purpose

In today’s digital age, users often seek emotional support through chat interfaces. This project:

- Understands users' emotional expressions using AI
- Responds with context-aware, comforting messages
- Detects mental health crises (e.g., suicidal ideation)
- Guides users to trusted emotional wellness resources
- Offers a lightweight backend for integration with any UI

---

##  Core Architecture

### 🔹 1. Emotion Classification (Transformer Model)

The model used is:

> [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)

It predicts 15+ emotions, including:
``joy``, ``sadness``, ``anger``, ``fear``, ``surprise``, ``love``, ``gratitude``, ``relief``, ``neutral``

Code:
```python
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
logits = outputs.logits
probs = F.softmax(logits, dim=1)
```

---

### 🔹 2. Crisis Detection (Safety Filter)

The app uses deterministic keyword scanning to flag high-risk phrases:
```python
["suicide", "kill myself", "end it", "hurt myself", "want to die"]
```

If detected:
- Bypasses model
- Returns a “crisis” emotion
- Offers suicide prevention links
- Logs a warning (without storing sensitive data)

---

### 🔹 3. Rate Limiting (Anti-Spam)

IP-based rate control prevents overuse:
- Max: **1 request per 2 seconds per IP**
- Uses a `defaultdict` with timestamps
- Returns "rate_limited" message if exceeded

---

### 🔹 4. Prewritten Emotional Responses

Each emotion maps to a **pool of comforting, conversational replies**:
```python
"anger": [
  "Anger is a valid feeling. Do you want to explore what's behind it?",
  ...
]
```

Messages are curated to:
- Normalize emotions
- Encourage reflection
- Avoid judgment or advice
- Promote open conversation

---

### 🔹 5. Emotion-Based Resource Linking

Each emotion is linked to a mental health page:
```python
"joy" → "https://www.helpguide.org/articles/mental-health/positive-psychology.htm"
"crisis" → "https://suicidepreventionlifeline.org/"
```

Sources include:
- mentalhealth.gov
- helpguide.org
- psychologytoday.com
- psychcentral.com

---

## 🔄 API Workflow

### ✅ POST `/chat`

**Sample Request:**
```json
{
  "message": "I'm feeling really hopeless and tired of everything."
}
```

**Internal flow:**
1. Check rate limit for IP
2. Detect crisis phrases
3. Run text through transformer
4. Apply confidence threshold (`0.4`)
5. Select emotion-specific response
6. Return message, emotion, confidence, resource, timestamp

**Response Example:**
```json
{
  "response": "I'm really sorry you're feeling this way. Want to talk more about it?",
  "emotion": "sadness",
  "confidence": "0.91",
  "resource": "https://www.mentalhealth.gov/talk/feelings/sadness",
  "timestamp": "2025-07-19T09:30:00Z"
}
```

---

## 📦 Tech Stack

- **Backend**: Flask (Python)
- **ML Model**: Hugging Face Transformers
- **Libraries**: `torch`, `transformers`, `flask-cors`
- **Hosting-ready**: Works with Gunicorn, Docker, Heroku, etc.
- **Frontend-friendly**: CORS enabled; plug into React, Vue, or mobile apps

---

## 🛠️ Setup Instructions

```bash
git clone https://github.com/your-username/emotion-chat-api.git
cd emotion-chat-api
pip install -r requirements.txt
python app.py
```

---

## 📁 Project Structure

```
emotion-chat-api/
├── app.py              # Main Flask server
├── templates/
│   └── index.html      # Optional UI for testing
├── requirements.txt    # Python packages
└── README.md           # Project documentation
```

---

## 🛡️ Ethics & Limitations

- ❗ **Not a medical tool** – designed for early support, not diagnosis
- ⚠️ Crisis detection is rule-based for 100% reliability
- ✅ Anonymous use (no login or personal data stored)
- 👂 Focus on listening, not giving advice
- 🧠 Can support emotional journaling, check-ins, and wellness apps

---

## ✨ Future Enhancements

- Sentiment/emotion trend graph
- GPT-based generative responses
- Speech or voice-based emotion classification
- Multilingual support
- Database-backed message logging (optional)

---

## 🙌 Acknowledgements

- Hugging Face community for models
- Torch and Flask open source teams
- Mental health professionals for curated message suggestions

---

## 📜 License

**MIT License**  
Feel free to use, share, and modify with proper attribution.

---

## 📣 Contributions Welcome

This project is open to enhancements from developers, mental health professionals, and researchers. Suggestions for improving ethical safeguards, message design, or multilingual support are encouraged.

```bash
# Fork this repository
# Create your feature branch: git checkout -b feature/your-idea
# Commit your changes: git commit -am 'Add awesome idea'
# Push to the branch: git push origin feature/your-idea
# Create a Pull Request 🚀
```

---

> Built with care to promote mental wellness through empathetic technology.
