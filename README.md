# 🎤 SpeechFindr
> **AI-Powered Speech Recognition Web Application**  
> Final Year Project · Iqra University, Karachi · B.S. Information Technology · 2024–2025

---

## 🌐 Live Pages
| Page | File | Description |
|------|------|-------------|
| Home | `index.html` | Landing page with hero, features, team |
| App | `project.html` | Core AI application interface |
| Contact | `contact.html` | Team & project contact info |

---

## 📁 Project Structure
```
SpeechFindr/
├── index.html              ← Landing Page
├── project.html            ← Core App UI
├── contact.html            ← Contact Page
├── assets/
│   ├── css/
│   │   └── style.css       ← Global Design System
│   ├── js/                 ← (reserved for future modules)
│   └── images/             ← (reserved for team photos)
└── README.md
```

---

## 🎨 Design System

| Token | Value | Usage |
|---|---|---|
| Background | `#050C1A` | Page base |
| Surface | `rgba(255,255,255,0.035)` | Cards |
| Blue | `#3B82F6` | Primary accent |
| Cyan | `#06B6D4` | Secondary accent |
| Sky | `#0EA5E9` | Gradient end |
| Text | `#F0F6FF` | Primary text |
| Muted | `#7E9BB5` | Secondary text |
| Font Head | Plus Jakarta Sans | Headings |
| Font Body | Space Grotesk | Body text |

**Theme:** Electric Blue · Dark Mode · Glassmorphism · Aurora Mesh

---

## 🚀 How to Run
```bash
# No build step required — pure HTML/CSS/JS
# Just open index.html in any modern browser
open index.html
```

### Run AI Backend (Groq + YouTube fallback)
```bash
cd backend
pip install -r requirements.txt

# Option A: set env in PowerShell
$env:GROQ_API_KEY=""
$env:GROQ_SUMMARY_API_KEY="your_summary_key_here"
$env:YTDLP_COOKIES_FROM_BROWSER=""
# Optional if your network blocks YouTube:
# $env:YTDLP_PROXY="http://127.0.0.1:7890"

# Option B: create backend/.env from backend/.env.example
# GROQ_API_KEY=your_new_rotated_key_here
# GROQ_SUMMARY_API_KEY=your_summary_key_here

uvicorn app:app --reload --host 127.0.0.1 --port 8001
```

- Frontend calls `http://127.0.0.1:8001/youtube/transcript`.
- Uploaded files use `http://127.0.0.1:8001/file/transcript`.
- Summary uses `http://127.0.0.1:8001/summary`.
- Flow:
  1) Try YouTube native captions in selected language
  2) Try YouTube auto-captions fallback
  3) If unavailable, auto-transcribe using Groq Whisper
  4) For uploaded video/audio files, extract audio and transcribe in parallel Groq chunks
  5) Re-processing the same uploaded file/language uses backend cache
  6) Generate AI summaries (general + keyword-focused) from transcript text

---

## 🛠 Tech Stack

### Frontend (This Repo)
- Vanilla HTML5 / CSS3 / JavaScript ES6+
- Three.js r128 — Aurora background & particle network
- Google Fonts — Plus Jakarta Sans + Space Grotesk
- CSS Custom Properties — full design token system
- IntersectionObserver — scroll-triggered animations
- Custom cursor with smooth ring tracking

### Backend (Python — connect separately)
| Tool | Purpose |
|---|---|
| OpenAI Whisper | Speech-to-text transcription |
| yt-dlp | YouTube audio extraction |
| FastAPI | REST API server |
| FFmpeg | Audio processing pipeline |
| HuggingFace Transformers | NLP summarization |
| Google Translate API | 50+ language translation |
| gTTS | Text-to-speech audio output |

---

## 👥 Team
| Name | Role | GitHub | LinkedIn |
|---|---|---|---|
| **Anas Tanveer** | Lead · Backend | [anastanveer653](https://github.com/anastanveer653) | [anastanveer-it](https://linkedin.com/in/anastanveer-it/) |
| Member 2 | AI · NLP | — | — |
| Member 3 | Frontend | — | — |
| Member 4 | Research | — | — |

---

## 🎓 Academic Info
- **University:** Iqra University, Karachi  
- **Degree:** B.S. Information Technology  
- **Year:** 2024–2025  
- **Supervisor:** [Supervisor Name]  

---

*Built with ❤️ — SpeechFindr Team, Iqra University Karachi*
