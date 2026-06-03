import os
import re
import subprocess
import tempfile
import time
import io
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import List, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from gtts import gTTS
from groq import Groq
from pydantic import BaseModel
from yt_dlp import YoutubeDL


API_TITLE = "SpeechFindr API"
DEFAULT_MODEL = "whisper-large-v3-turbo"
CACHE_TTL_SECONDS = 3600
HTTP_RETRIES = 3
UPLOAD_CHUNK_SECONDS = 120
UPLOAD_MAX_PARALLEL = 6
load_dotenv()
TRANSCRIPT_CACHE: dict = {}
TRANSLATION_TOGGLE = 0
TRANSLATION_LOCK = Lock()


class YouTubeTranscriptRequest(BaseModel):
    url: str
    language: str = "auto"


class TranscriptSegment(BaseModel):
    t: str
    s: int
    tx: str


class YouTubeTranscriptResponse(BaseModel):
    source: str
    language: str
    segments: List[TranscriptSegment]


class SummaryRequest(BaseModel):
    transcript: str
    mode: str = "general"
    keyword: str = ""
    length: str = "medium"
    language: str = "en"


class SummaryResponse(BaseModel):
    summary: str
    mode: str
    keyword: str
    length: str
    language: str


class TopicRequest(BaseModel):
    transcript: str
    max_topics: int = 6


class TopicResponse(BaseModel):
    topics: List[str]


class ChapterSegmentInput(BaseModel):
    s: int
    tx: str


class ChapterRequest(BaseModel):
    segments: List[ChapterSegmentInput]
    duration_seconds: Optional[int] = None
    max_chapters: int = 6


class ChapterItem(BaseModel):
    start: int
    end: int
    start_t: str
    end_t: str
    title: str


class ChapterResponse(BaseModel):
    chapters: List[ChapterItem]


class QARequest(BaseModel):
    segments: List[ChapterSegmentInput]
    question: str
    max_context: int = 12
    history: List[dict] = []


class QAResponse(BaseModel):
    answer: str
    timestamp_s: int
    timestamp_t: str
    evidence: List[dict]


class TranslationSegmentInput(BaseModel):
    t: str
    s: int
    tx: str


class TranslateRequest(BaseModel):
    segments: List[TranslationSegmentInput]
    target_language: str


class TranslateResponse(BaseModel):
    translated_text: str
    translated_segments: List[TranslationSegmentInput]


class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    voice: str = "neutral"  # neutral | male | female


app = FastAPI(title=API_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def fmt(seconds: float) -> str:
    value = max(0, int(seconds))
    return f"{value // 60}:{value % 60:02d}"


def parse_vtt(vtt: str) -> List[dict]:
    lines = vtt.splitlines()
    segments: List[dict] = []
    i = 0
    while i < len(lines):
        line = (lines[i] or "").strip()
        if "-->" in line:
            start_raw = line.split(" --> ")[0].strip()
            sec = to_seconds(start_raw)
            text_lines: List[str] = []
            i += 1
            while i < len(lines):
                cue = (lines[i] or "").strip()
                if not cue:
                    break
                text_lines.append(re.sub(r"<[^>]*>", "", cue))
                i += 1
            text = " ".join(text_lines).strip()
            if text:
                segments.append({"t": fmt(sec), "s": int(sec), "tx": text})
        i += 1
    return segments


def to_seconds(raw: str) -> float:
    parts = raw.split(":")
    if len(parts) != 3:
        return 0.0
    try:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2].replace(",", "."))
    except ValueError:
        return 0.0


def extract_video_id(url: str) -> Optional[str]:
    m = re.search(r"(?:v=|youtu\.be/|embed/|shorts/|live/)([a-zA-Z0-9_-]{11})", url)
    return m.group(1) if m else None


def fetch_youtube_captions(video_id: str, wanted_language: str) -> Optional[List[dict]]:
    lang = (wanted_language or "auto").lower()
    candidates: List[str] = []
    if lang != "auto":
        candidates.extend(
            [
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang={lang}&fmt=vtt",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang={lang}&kind=asr&fmt=vtt",
            ]
        )
    candidates.extend(
        [
            f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&fmt=vtt",
            f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&kind=asr&fmt=vtt",
        ]
    )
    for url in candidates:
        for _ in range(HTTP_RETRIES):
            try:
                resp = requests.get(url, timeout=20, headers={"User-Agent": os.getenv("HTTP_USER_AGENT", "Mozilla/5.0")})
            except requests.RequestException:
                time.sleep(0.35)
                continue
            if not resp.ok:
                time.sleep(0.35)
                continue
            segments = parse_vtt(resp.text)
            if segments:
                return segments
            time.sleep(0.2)
    return None


def _lang_priority_keys(wanted_language: str, available_keys: List[str]) -> List[str]:
    wanted = (wanted_language or "auto").lower()
    norm_map = {k.lower(): k for k in available_keys}
    order: List[str] = []

    if wanted != "auto":
        exact = norm_map.get(wanted)
        if exact:
            order.append(exact)
        prefixed = [orig for key, orig in norm_map.items() if key.startswith(f"{wanted}-")]
        order.extend([k for k in prefixed if k not in order])

    for fallback in ("en",):
        exact = norm_map.get(fallback)
        if exact and exact not in order:
            order.append(exact)
        prefixed = [orig for key, orig in norm_map.items() if key.startswith(f"{fallback}-")]
        order.extend([k for k in prefixed if k not in order])

    for key in available_keys:
        if key not in order:
            order.append(key)
    return order


def _fetch_track_segments(track_list: List[dict]) -> Optional[List[dict]]:
    # Prefer vtt tracks first because parser is optimized for VTT.
    for ext_pref in ("vtt", None):
        for track in track_list:
            if ext_pref and (track.get("ext") or "").lower() != ext_pref:
                continue
            url = track.get("url")
            if not url:
                continue
            for _ in range(HTTP_RETRIES):
                try:
                    resp = requests.get(url, timeout=25, headers={"User-Agent": os.getenv("HTTP_USER_AGENT", "Mozilla/5.0")})
                except requests.RequestException:
                    time.sleep(0.35)
                    continue
                if not resp.ok:
                    time.sleep(0.35)
                    continue
                segments = parse_vtt(resp.text)
                if segments:
                    return segments
                time.sleep(0.2)
    return None


def fetch_youtube_captions_with_ytdlp(url: str, wanted_language: str) -> Optional[List[dict]]:
    browser_cookie = (os.getenv("YTDLP_COOKIES_FROM_BROWSER") or "").strip().lower()
    proxy = (os.getenv("YTDLP_PROXY") or "").strip()
    def _build_opts(use_cookie: bool) -> dict:
        opts = {
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
            "retries": 5,
            "fragment_retries": 5,
            "socket_timeout": 30,
        }
        if use_cookie and browser_cookie:
            opts["cookiesfrombrowser"] = (browser_cookie,)
        if proxy:
            opts["proxy"] = proxy
        return opts

    info = None
    attempts = [True, False] if browser_cookie else [False]
    for use_cookie in attempts:
        try:
            with YoutubeDL(_build_opts(use_cookie)) as ydl:
                info = ydl.extract_info(url, download=False)
            break
        except Exception:
            info = None
            continue
    if not info:
        return None

    subtitles = info.get("subtitles") or {}
    automatic = info.get("automatic_captions") or {}

    for source in (subtitles, automatic):
        if not source:
            continue
        keys = list(source.keys())
        for lang_key in _lang_priority_keys(wanted_language, keys):
            tracks = source.get(lang_key) or []
            segments = _fetch_track_segments(tracks)
            if segments:
                return segments
    return None


def translate_segments_if_needed(client: Groq, segments: List[dict], target_language: str) -> List[dict]:
    lang = (target_language or "auto").lower()
    if lang in ("auto", "en"):
        return segments

    # Larger chunk -> fewer API calls -> faster overall translation.
    # Prompt + parsing are strict to avoid truncation/missing-line issues.
    chunk_size = 50
    translated: List[dict] = []
    for i in range(0, len(segments), chunk_size):
        chunk = segments[i : i + chunk_size]
        lines = "\n".join([f"[{idx}] {seg['tx']}" for idx, seg in enumerate(chunk)])

        # Pick alternating keys per chunk so we don't overload a single translation resource.
        client_chunk = _pick_translation_groq_client()

        # Output is one line per input line, so allow enough tokens to avoid truncation.
        # (Hard cap for safety.)
        max_tokens = min(2500, 350 + (len(chunk) * 60))

        completion = client_chunk.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Translate each numbered line to the requested language."
                        " Output must contain EXACTLY the same numbering as input."
                        " For each i from 0 to N-1 output a single line:"
                        " [i] translated text"
                        " Do not output bullet points, explanations, or any extra text."
                        " Do not leave any line untranslated."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Target language code: {lang}\n\n{lines}",
                },
            ],
            top_p=1,
            max_completion_tokens=max_tokens,
        )
        raw = completion.choices[0].message.content or ""
        mapped: dict[int, str] = {}

        # Be tolerant to formatting; we just need bracketed line indices.
        index_re = re.compile(r"\[(\d+)\]\s*(.*)$")
        for row in (raw.splitlines() or []):
            line = (row or "").strip()
            if not line:
                continue
            m = index_re.search(line)
            if not m:
                continue
            idx = int(m.group(1))
            text = (m.group(2) or "").strip()
            if text:
                mapped[idx] = text

        expected = len(chunk)
        if len(mapped) < expected:
            # Recover missing/failed lines with a smaller second pass.
            missing_idxs = [idx for idx in range(expected) if not mapped.get(idx)]
            if missing_idxs:
                missing_segments = [chunk[idx] for idx in missing_idxs]
                recovered = translate_segments_if_needed(client, missing_segments, target_language)
                for missing_idx, rec_seg in zip(missing_idxs, recovered):
                    mapped[missing_idx] = rec_seg.get("tx") or mapped.get(missing_idx) or ""

        for idx, seg in enumerate(chunk):
            translated_text = mapped.get(idx) or seg["tx"]
            translated.append({"t": seg["t"], "s": seg["s"], "tx": translated_text})
    return translated


def _chunk_plain_transcript(text: str, duration_seconds: Optional[float] = None) -> List[dict]:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return []

    pieces = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
    if not pieces:
        pieces = [cleaned]

    merged: List[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip() if current else piece
        if current and len(candidate) > 140:
            merged.append(current)
            current = piece
        else:
            current = candidate
    if current:
        merged.append(current)

    total_chunks = max(1, len(merged))
    total_duration = max(float(duration_seconds or 0), float(total_chunks * 6))
    step = max(1, int(total_duration / total_chunks))

    segments: List[dict] = []
    for idx, chunk in enumerate(merged):
        start = min(int(idx * step), int(max(total_duration - 1, 0)))
        segments.append({"t": fmt(start), "s": start, "tx": chunk})
    return segments


def _segments_from_groq_result(result, duration_seconds: Optional[float] = None) -> List[dict]:
    segments: List[dict] = []
    raw_segments = getattr(result, "segments", None) or []
    for seg in raw_segments:
        text = (getattr(seg, "text", "") or "").strip()
        start = int(float(getattr(seg, "start", 0) or 0))
        if text:
            segments.append({"t": fmt(start), "s": start, "tx": text})

    if not segments:
        fallback_text = (getattr(result, "text", "") or "").strip()
        if fallback_text:
            segments = _chunk_plain_transcript(fallback_text, duration_seconds)

    if not segments:
        raise HTTPException(status_code=500, detail="Transcription returned no segments.")
    return segments


def _offset_segments(segments: List[dict], offset_seconds: int) -> List[dict]:
    adjusted: List[dict] = []
    for seg in segments:
        start = max(0, int(seg["s"]) + int(offset_seconds))
        adjusted.append({"t": fmt(start), "s": start, "tx": seg["tx"]})
    return adjusted


def _groq_client() -> Groq:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured.")
    return Groq(api_key=key)


def _summary_groq_client() -> Groq:
    key = os.getenv("GROQ_SUMMARY_API_KEY", "").strip()
    if not key:
        raise HTTPException(status_code=500, detail="GROQ_SUMMARY_API_KEY is not configured.")
    return Groq(api_key=key)


def _analysis_groq_client() -> Groq:
    key = (os.getenv("GROQ_ANALYSIS_API_KEY") or os.getenv("GROQ_API_KEY") or "").strip()
    if not key:
        raise HTTPException(status_code=500, detail="GROQ_ANALYSIS_API_KEY (or GROQ_API_KEY) is not configured.")
    return Groq(api_key=key)


def _translation_groq_client_1() -> Groq:
    key = (os.getenv("GROQ_TRANSLATION_API_KEY_1") or "").strip()
    if not key:
        raise HTTPException(status_code=500, detail="GROQ_TRANSLATION_API_KEY_1 is not configured.")
    return Groq(api_key=key)


def _translation_groq_client_2() -> Groq:
    key = (os.getenv("GROQ_TRANSLATION_API_KEY_2") or "").strip()
    if not key:
        raise HTTPException(status_code=500, detail="GROQ_TRANSLATION_API_KEY_2 is not configured.")
    return Groq(api_key=key)


def _pick_translation_groq_client() -> Groq:
    global TRANSLATION_TOGGLE
    with TRANSLATION_LOCK:
        TRANSLATION_TOGGLE += 1
        use_first = (TRANSLATION_TOGGLE % 2) == 1
    return _translation_groq_client_1() if use_first else _translation_groq_client_2()


def _is_groq_rate_limit_error(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    return "ratelimit" in name or "rate limit" in msg or "rate_limit_exceeded" in msg or "429" in msg


def _retry_sleep_seconds_from_exc(exc: Exception, attempt: int) -> float:
    # Groq usually includes: "Please try again in 1.38s"
    msg = str(exc)
    m = re.search(r"try again in\s*([0-9.]+)s", msg, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # Fallback exponential backoff
    return min(20.0, 1.5 * (attempt + 1))


def translate_segments_with_retries(client: Groq, segments: List[dict], target_language: str, max_attempts: int = 3) -> List[dict]:
    last_exc: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            return translate_segments_if_needed(client, segments, target_language)
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts - 1 or not _is_groq_rate_limit_error(exc):
                raise
            sleep_s = _retry_sleep_seconds_from_exc(exc, attempt)
            time.sleep(sleep_s)
    # Should be unreachable because we either return or raise above.
    raise HTTPException(status_code=429, detail=f"Translation failed due to rate limiting: {last_exc}")


def _tts_lang(code: str) -> str:
    c = (code or "en").strip().lower()
    # gTTS uses some specific codes; keep minimal mapping.
    if c == "zh":
        return "zh-CN"
    if c in {"pt", "pt-br"}:
        return "pt"
    return c


def _tts_voice_params(language: str, voice: str) -> dict:
    """
    gTTS does not offer true male/female voices. We provide a best-effort
    "voice" option by varying pacing and (for English only) tld accent.
    """
    v = (voice or "neutral").strip().lower()
    lang = (language or "en").strip().lower()
    params: dict = {"slow": False}

    if v == "female":
        params["slow"] = True
    elif v == "male":
        params["slow"] = False

    # Accent variation only meaningfully affects English. Keep safe defaults elsewhere.
    if lang.startswith("en"):
        if v == "female":
            params["tld"] = "co.uk"
        elif v == "male":
            params["tld"] = "com"
        else:
            params["tld"] = "com.au"
    return params


def _filter_transcript_for_keyword(transcript: str, keyword: str) -> str:
    kw = (keyword or "").strip().lower()
    if not kw:
        return transcript
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", transcript or "") if s.strip()]
    matched = [s for s in sentences if kw in s.lower()]
    return " ".join(matched)


def generate_summary(
    transcript: str,
    mode: str = "general",
    keyword: str = "",
    length: str = "medium",
    language: str = "en",
) -> str:
    normalized_mode = (mode or "general").strip().lower()
    normalized_length = (length or "medium").strip().lower()
    normalized_language = (language or "en").strip().lower()
    raw_text = (transcript or "").strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Transcript is empty.")
    if normalized_mode not in {"general", "keyword"}:
        raise HTTPException(status_code=400, detail="Invalid summary mode. Use 'general' or 'keyword'.")
    if normalized_length not in {"short", "medium", "detailed"}:
        raise HTTPException(status_code=400, detail="Invalid summary length. Use 'short', 'medium', or 'detailed'.")
    if normalized_language not in {"en", "ur", "ar"}:
        raise HTTPException(status_code=400, detail="Invalid summary language. Use 'en', 'ur', or 'ar'.")

    length_words = {"short": 50, "medium": 150, "detailed": 300}
    target_words = length_words[normalized_length]
    language_names = {"en": "English", "ur": "Urdu", "ar": "Arabic"}
    language_name = language_names[normalized_language]

    text_for_model = raw_text
    prompt = f"Summarize this video transcript in around {target_words} words."
    normalized_keyword = (keyword or "").strip()
    if normalized_mode == "keyword":
        if not normalized_keyword:
            raise HTTPException(status_code=400, detail="Keyword is required for keyword summary.")
        filtered = _filter_transcript_for_keyword(raw_text, normalized_keyword)
        if not filtered:
            return f"No transcript lines found for keyword '{normalized_keyword}'."
        text_for_model = filtered
        prompt = (
            f"Summarize only the parts about '{normalized_keyword}' from this transcript "
            f"in around {target_words} words."
        )

    # Keep prompt size bounded so very long transcripts do not overflow model context.
    if len(text_for_model) > 28000:
        text_for_model = text_for_model[:28000]

    cache_key = (
        f"summary:{hashlib.sha256(text_for_model.encode('utf-8')).hexdigest()}:"
        f"{normalized_mode}:{normalized_keyword.lower()}:{normalized_length}:{normalized_language}"
    )
    now = time.time()
    cached = TRANSCRIPT_CACHE.get(cache_key)
    if cached and (now - cached["ts"]) < CACHE_TTL_SECONDS:
        return cached["payload"]["summary"]

    client = _summary_groq_client()
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You summarize transcripts. Keep output factual and clear. "
                    f"Target approximately {target_words} words. "
                    f"Output language must be {language_name}. "
                    "Do not include bullet points."
                ),
            },
            {
                "role": "user",
                "content": f"{prompt}\n\nTranscript:\n{text_for_model}",
            },
        ],
        temperature=0.3,
        max_completion_tokens=520,
        top_p=1,
    )
    summary = (completion.choices[0].message.content or "").strip()
    if not summary:
        raise HTTPException(status_code=502, detail="Summary generation returned empty output.")
    words = summary.split()
    if len(words) > int(target_words * 1.5):
        summary = " ".join(words[: int(target_words * 1.5)])
    TRANSCRIPT_CACHE[cache_key] = {"ts": now, "payload": {"summary": summary}}
    return summary


def detect_topics(transcript: str, max_topics: int = 6) -> List[str]:
    text = (transcript or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Transcript is empty.")

    limit = max(3, min(int(max_topics or 6), 12))
    text_for_model = text[:22000] if len(text) > 22000 else text
    cache_key = f"topics:{hashlib.sha256(text_for_model.encode('utf-8')).hexdigest()}:{limit}"
    now = time.time()
    cached = TRANSCRIPT_CACHE.get(cache_key)
    if cached and (now - cached["ts"]) < CACHE_TTL_SECONDS:
        return cached["payload"]["topics"]

    client = _analysis_groq_client()
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract concise topic tags from transcript text. "
                    "Return ONLY a JSON array of short strings with no explanations."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Find the top {limit} topics in this transcript. "
                    "Each topic should be 1-3 words.\n\n"
                    f"Transcript:\n{text_for_model}"
                ),
            },
        ],
        temperature=0.1,
        max_completion_tokens=220,
        top_p=1,
    )
    raw = (completion.choices[0].message.content or "").strip()
    topics: List[str] = []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            topics = [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        tokens = re.split(r"[,|\n]+", raw)
        topics = [t.strip(" -•\t\r\"'") for t in tokens if t.strip(" -•\t\r\"'")]

    normalized: List[str] = []
    seen = set()
    for topic in topics:
        cleaned = re.sub(r"\s+", " ", topic).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
        if len(normalized) >= limit:
            break

    TRANSCRIPT_CACHE[cache_key] = {"ts": now, "payload": {"topics": normalized}}
    return normalized


def detect_chapters(
    segments: List[ChapterSegmentInput],
    duration_seconds: Optional[int] = None,
    max_chapters: int = 6,
) -> List[dict]:
    clean_segments = [
        {"s": max(0, int(seg.s)), "tx": (seg.tx or "").strip()}
        for seg in segments
        if (seg.tx or "").strip()
    ]
    if not clean_segments:
        raise HTTPException(status_code=400, detail="Transcript segments are empty.")

    clean_segments.sort(key=lambda x: x["s"])
    max_items = max(40, min(len(clean_segments), 260))
    sampled = clean_segments[:max_items]
    transcript_for_model = "\n".join([f"[{fmt(item['s'])}] {item['tx']}" for item in sampled])

    chapter_limit = max(3, min(int(max_chapters or 6), 12))
    effective_duration = int(duration_seconds or clean_segments[-1]["s"] + 30)
    cache_key = (
        f"chapters:{hashlib.sha256(transcript_for_model.encode('utf-8')).hexdigest()}:"
        f"{chapter_limit}:{effective_duration}"
    )
    now = time.time()
    cached = TRANSCRIPT_CACHE.get(cache_key)
    if cached and (now - cached["ts"]) < CACHE_TTL_SECONDS:
        return cached["payload"]["chapters"]

    client = _analysis_groq_client()
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You create chapter markers for transcripts. "
                    "Return ONLY a JSON array. Each item must be "
                    '{"start":"MM:SS","title":"short chapter title"}.'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Create up to {chapter_limit} chapter markers from this transcript.\n"
                    "Rules: chapter titles 2-5 words, start times in ascending order, no overlaps.\n\n"
                    f"Transcript:\n{transcript_for_model}"
                ),
            },
        ],
        temperature=0.1,
        max_completion_tokens=420,
        top_p=1,
    )
    raw = (completion.choices[0].message.content or "").strip()

    parsed_items: List[dict] = []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    parsed_items.append(item)
    except Exception:
        parsed_items = []

    starts: List[dict] = []
    for item in parsed_items:
        start_raw = str(item.get("start", "")).strip()
        title = re.sub(r"\s+", " ", str(item.get("title", "")).strip())
        if not title:
            continue
        m = re.match(r"^(\d{1,2}):(\d{2})$", start_raw)
        if not m:
            continue
        sec = int(m.group(1)) * 60 + int(m.group(2))
        if sec < 0:
            continue
        starts.append({"start": sec, "title": title})

    if not starts:
        # Fallback without LLM parse: split timeline evenly.
        approx = max(3, min(chapter_limit, 6))
        step = max(60, int(effective_duration / approx))
        starts = [{"start": i * step, "title": f"Chapter {i+1}"} for i in range(approx)]

    starts = sorted(starts, key=lambda x: x["start"])
    deduped: List[dict] = []
    seen_starts = set()
    for row in starts:
        s = min(max(0, int(row["start"])), max(0, effective_duration - 1))
        if s in seen_starts:
            continue
        seen_starts.add(s)
        deduped.append({"start": s, "title": row["title"]})
        if len(deduped) >= chapter_limit:
            break

    chapters: List[dict] = []
    for idx, row in enumerate(deduped):
        start = row["start"]
        end = deduped[idx + 1]["start"] if idx + 1 < len(deduped) else effective_duration
        if end <= start:
            end = min(effective_duration, start + 60)
        chapters.append(
            {
                "start": start,
                "end": end,
                "start_t": fmt(start),
                "end_t": fmt(end),
                "title": row["title"],
            }
        )

    TRANSCRIPT_CACHE[cache_key] = {"ts": now, "payload": {"chapters": chapters}}
    return chapters


def _tokenize(text: str) -> List[str]:
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", (text or "").lower())
    return [w for w in t.split() if len(w) >= 3]


def answer_question(
    segments: List[ChapterSegmentInput],
    question: str,
    max_context: int = 12,
    history: Optional[List[dict]] = None,
) -> dict:
    q = (question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question is empty.")

    clean = [
        {"s": max(0, int(seg.s)), "tx": (seg.tx or "").strip()}
        for seg in segments
        if (seg.tx or "").strip()
    ]
    if not clean:
        raise HTTPException(status_code=400, detail="Transcript segments are empty.")

    q_tokens = _tokenize(q)
    q_counts = Counter(q_tokens)

    scored = []
    for row in clean:
        tokens = _tokenize(row["tx"])
        if not tokens:
            continue
        counts = Counter(tokens)
        score = 0
        for tok, wt in q_counts.items():
            if tok in counts:
                score += wt * min(3, counts[tok])
        if score > 0:
            scored.append((score, row["s"], row["tx"]))

    scored.sort(key=lambda x: (-x[0], x[1]))
    k = max(6, min(int(max_context or 12), 24))
    top = scored[:k] if scored else [(0, r["s"], r["tx"]) for r in clean[:k]]

    context_lines = "\n".join([f"[{fmt(s)}] {tx}" for _, s, tx in top])
    history = history or []
    # Keep only last few turns to bound prompt size
    history_trimmed = history[-6:] if len(history) > 6 else history
    history_text = "\n".join(
        [
            f"{('User' if (m.get('role')=='user') else 'Assistant')}: {str(m.get('content','')).strip()}"
            for m in history_trimmed
            if str(m.get("content", "")).strip()
        ]
    )

    cache_key = f"qa:{hashlib.sha256((q + '|' + context_lines + '|' + history_text).encode('utf-8')).hexdigest()}"
    now = time.time()
    cached = TRANSCRIPT_CACHE.get(cache_key)
    if cached and (now - cached["ts"]) < CACHE_TTL_SECONDS:
        return cached["payload"]

    client = _analysis_groq_client()
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions about a video using ONLY the provided transcript excerpts. "
                    "If unsure, say you cannot find it in the provided transcript. "
                    "Always include one best timestamp in the format 'Timestamp: MM:SS'. "
                    "Keep the answer concise (2-5 sentences)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Conversation so far (may be empty):\n{history_text}\n\n"
                    f"Question: {q}\n\nTranscript excerpts:\n{context_lines}"
                ),
            },
        ],
        temperature=0.2,
        max_completion_tokens=220,
        top_p=1,
    )
    answer = (completion.choices[0].message.content or "").strip()
    if not answer:
        raise HTTPException(status_code=502, detail="Q&A returned empty output.")

    m = re.search(r"Timestamp:\s*(\d{1,2}):(\d{2})", answer)
    ts_s = top[0][1] if top else 0
    if m:
        ts_s = int(m.group(1)) * 60 + int(m.group(2))

    evidence = [{"s": s, "t": fmt(s), "tx": tx} for _, s, tx in top[: min(6, len(top))]]
    payload = {"answer": answer, "timestamp_s": int(ts_s), "timestamp_t": fmt(ts_s), "evidence": evidence}
    TRANSCRIPT_CACHE[cache_key] = {"ts": now, "payload": payload}
    return payload


def _run_ffmpeg(command: List[str], timeout: int, failure_detail: str) -> None:
    try:
        proc = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="FFmpeg is not installed or not available in PATH.",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail=failure_detail) from exc

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        detail = stderr.splitlines()[-1] if stderr else failure_detail
        raise HTTPException(status_code=400, detail=detail)


def _write_uploaded_audio(filename: str, data: bytes, temp_dir: str) -> str:
    suffix = Path(filename or "upload.bin").suffix or ".bin"
    source_path = os.path.join(temp_dir, f"source{suffix}")
    audio_path = os.path.join(temp_dir, "audio.mp3")
    with open(source_path, "wb") as source_file:
        source_file.write(data)

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        source_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "48k",
        "-compression_level",
        "9",
        audio_path,
    ]
    _run_ffmpeg(
        ffmpeg_cmd,
        timeout=180,
        failure_detail="FFmpeg conversion timed out for uploaded media.",
    )
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=400, detail="Audio extraction failed: FFmpeg could not extract audio from this file.")
    return audio_path


def _slice_audio_chunk(audio_path: str, start_seconds: int, chunk_seconds: int, out_path: str) -> None:
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_seconds),
        "-t",
        str(chunk_seconds),
        "-i",
        audio_path,
        "-acodec",
        "copy",
        out_path,
    ]
    _run_ffmpeg(
        ffmpeg_cmd,
        timeout=120,
        failure_detail="FFmpeg timed out while slicing uploaded audio.",
    )


def _transcribe_audio_bytes(audio_name: str, audio_bytes: bytes, duration_seconds: Optional[float] = None) -> List[dict]:
    client = _groq_client()
    result = client.audio.transcriptions.create(
        file=(audio_name, audio_bytes),
        model=DEFAULT_MODEL,
        temperature=0,
        response_format="verbose_json",
        timestamp_granularities=["segment"],
    )
    return _segments_from_groq_result(result, duration_seconds)


def transcribe_youtube_with_groq(url: str, target_language: str) -> List[dict]:
    browser_cookie = (os.getenv("YTDLP_COOKIES_FROM_BROWSER") or "").strip().lower()
    proxy = (os.getenv("YTDLP_PROXY") or "").strip()
    client = _groq_client()
    with tempfile.TemporaryDirectory() as temp_dir:
        def _build_opts(use_cookie: bool) -> dict:
            opts = {
                "format": "bestaudio/best",
                "outtmpl": os.path.join(temp_dir, "audio.%(ext)s"),
                "noplaylist": True,
                "quiet": True,
                "no_warnings": True,
                "retries": 5,
                "fragment_retries": 5,
                "socket_timeout": 30,
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }
                ],
            }
            if use_cookie and browser_cookie:
                opts["cookiesfrombrowser"] = (browser_cookie,)
            if proxy:
                opts["proxy"] = proxy
            return opts

        last_error = None
        attempts = [True, False] if browser_cookie else [False]
        for use_cookie in attempts:
            try:
                with YoutubeDL(_build_opts(use_cookie)) as ydl:
                    ydl.download([url])
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise HTTPException(status_code=502, detail=f"YouTube audio download failed: {last_error}") from last_error

        audio_path = os.path.join(temp_dir, "audio.mp3")
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=500, detail="Audio extraction failed.")

        with open(audio_path, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), audio_file.read()),
                model=DEFAULT_MODEL,
                temperature=0,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        segments = _segments_from_groq_result(result)
        return translate_segments_if_needed(client, segments, target_language)


def transcribe_uploaded_media_with_groq(
    filename: str,
    data: bytes,
    target_language: str,
    duration_seconds: Optional[float] = None,
) -> List[dict]:
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    file_hash = hashlib.sha256(data).hexdigest()
    cache_key = f"upload:v2:{file_hash}:{(target_language or 'auto').lower()}:{int(float(duration_seconds or 0))}"
    now = time.time()
    cached = TRANSCRIPT_CACHE.get(cache_key)
    if cached and (now - cached["ts"]) < CACHE_TTL_SECONDS:
        return cached["payload"]["segments"]

    estimated_duration = int(float(duration_seconds or 0)) if duration_seconds else 0
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = _write_uploaded_audio(filename, data, temp_dir)
        if estimated_duration > UPLOAD_CHUNK_SECONDS:
            chunk_jobs = []
            start = 0
            while start < estimated_duration:
                chunk_len = min(UPLOAD_CHUNK_SECONDS, estimated_duration - start)
                chunk_path = os.path.join(temp_dir, f"chunk_{start}.mp3")
                _slice_audio_chunk(audio_path, start, chunk_len, chunk_path)
                chunk_jobs.append((start, chunk_len, chunk_path))
                start += UPLOAD_CHUNK_SECONDS

            def _work(job: tuple[int, int, str]) -> List[dict]:
                offset, chunk_len, chunk_path = job
                with open(chunk_path, "rb") as chunk_file:
                    raw_segments = _transcribe_audio_bytes(
                        os.path.basename(chunk_path),
                        chunk_file.read(),
                        chunk_len,
                    )
                return _offset_segments(raw_segments, offset)

            workers = max(1, min(UPLOAD_MAX_PARALLEL, len(chunk_jobs)))
            merged: List[dict] = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for part in pool.map(_work, chunk_jobs):
                    merged.extend(part)
            segments = sorted(merged, key=lambda seg: seg["s"])
        else:
            with open(audio_path, "rb") as audio_file:
                media_bytes = audio_file.read()
            segments = _transcribe_audio_bytes("upload.mp3", media_bytes, duration_seconds)

    client = _groq_client()
    translated = translate_segments_if_needed(client, segments, target_language)
    TRANSCRIPT_CACHE[cache_key] = {
        "ts": now,
        "payload": {"source": "groq_whisper", "language": target_language, "segments": translated},
    }
    return translated


@app.get("/health")
def health():
    return {"ok": True, "service": API_TITLE}


def resolve_youtube_transcript(url: str, language: str) -> YouTubeTranscriptResponse:
    video_id = extract_video_id(url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL.")
    cache_key = f"{video_id}:{(language or 'auto').lower()}"
    now = time.time()
    cached = TRANSCRIPT_CACHE.get(cache_key)
    if cached and (now - cached["ts"]) < CACHE_TTL_SECONDS:
        return cached["payload"]

    # Robust method first: inspect actual subtitle tracks from video metadata.
    try:
        captions = fetch_youtube_captions_with_ytdlp(url, language)
    except Exception:
        captions = None
    if captions:
        payload = {"source": "youtube_captions", "language": language, "segments": captions}
        TRANSCRIPT_CACHE[cache_key] = {"ts": now, "payload": payload}
        return payload

    # Fallback to direct timedtext endpoints.
    try:
        captions = fetch_youtube_captions(video_id, language)
    except Exception:
        captions = None
    if captions:
        payload = {"source": "youtube_captions", "language": language, "segments": captions}
        TRANSCRIPT_CACHE[cache_key] = {"ts": now, "payload": payload}
        return payload

    try:
        transcribed = transcribe_youtube_with_groq(url, language)
        payload = {"source": "groq_whisper", "language": language, "segments": transcribed}
        TRANSCRIPT_CACHE[cache_key] = {"ts": now, "payload": payload}
        return payload
    except HTTPException:
        return {"source": "unavailable", "language": language, "segments": []}
    except Exception:
        return {"source": "unavailable", "language": language, "segments": []}


@app.post("/youtube/transcript", response_model=YouTubeTranscriptResponse)
def youtube_transcript(payload: YouTubeTranscriptRequest):
    return resolve_youtube_transcript(payload.url, payload.language)


@app.get("/youtube/transcript", response_model=YouTubeTranscriptResponse)
def youtube_transcript_get(url: str, language: str = "auto"):
    return resolve_youtube_transcript(url, language)


@app.post("/file/transcript", response_model=YouTubeTranscriptResponse)
async def file_transcript(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    duration_seconds: Optional[float] = Form(None),
):
    try:
        data = await file.read()
        segments = transcribe_uploaded_media_with_groq(
            file.filename or "upload.webm",
            data,
            language,
            duration_seconds,
        )
        return {"source": "groq_whisper", "language": language, "segments": segments}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Uploaded media transcription failed: {exc}") from exc


@app.post("/summary", response_model=SummaryResponse)
def summarize_transcript(payload: SummaryRequest):
    normalized_length = (payload.length or "medium").lower()
    normalized_language = (payload.language or "en").lower()
    summary = generate_summary(payload.transcript, payload.mode, payload.keyword, normalized_length, normalized_language)
    return {
        "summary": summary,
        "mode": (payload.mode or "general").lower(),
        "keyword": payload.keyword or "",
        "length": normalized_length,
        "language": normalized_language,
    }


@app.post("/topics", response_model=TopicResponse)
def extract_topics(payload: TopicRequest):
    topics = detect_topics(payload.transcript, payload.max_topics)
    return {"topics": topics}


@app.post("/chapters", response_model=ChapterResponse)
def extract_chapters(payload: ChapterRequest):
    chapters = detect_chapters(payload.segments, payload.duration_seconds, payload.max_chapters)
    return {"chapters": chapters}


@app.post("/qa", response_model=QAResponse)
def qa_video(payload: QARequest):
    result = answer_question(payload.segments, payload.question, payload.max_context, payload.history)
    return result


@app.post("/translate", response_model=TranslateResponse)
def translate_endpoint(payload: TranslateRequest):
    target_language = (payload.target_language or "auto").strip().lower()
    if not payload.segments:
        raise HTTPException(status_code=400, detail="No segments provided for translation.")

    raw_text = " ".join([seg.tx for seg in payload.segments if seg.tx]).strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Transcript text is empty.")

    # Cache by transcript content + target language so repeated clicks don't re-burn tokens.
    cache_key = f"translate:{hashlib.sha256(raw_text.encode('utf-8')).hexdigest()}:{target_language}"
    now = time.time()
    cached = TRANSCRIPT_CACHE.get(cache_key)
    if cached and (now - cached["ts"]) < CACHE_TTL_SECONDS:
        return cached["payload"]

    segments = [{"t": seg.t, "s": seg.s, "tx": seg.tx} for seg in payload.segments if seg.tx]
    client = _pick_translation_groq_client()
    translated_segments = translate_segments_with_retries(client, segments, target_language)

    translated_text = " ".join([s.get("tx", "") for s in translated_segments]).strip()
    payload_out: TranslateResponse = TranslateResponse(translated_text=translated_text, translated_segments=translated_segments)  # type: ignore[arg-type]

    TRANSCRIPT_CACHE[cache_key] = {"ts": now, "payload": payload_out}
    return payload_out


@app.post("/tts")
def tts_endpoint(payload: TTSRequest):
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty.")

    language = _tts_lang(payload.language)
    voice = (payload.voice or "neutral").strip().lower()
    if voice not in {"neutral", "male", "female"}:
        raise HTTPException(status_code=400, detail="Invalid voice. Use neutral, male, or female.")

    # Bound size so we don't send extremely large payloads to gTTS in one go.
    if len(text) > 12000:
        text = text[:12000]

    key = f"tts:{hashlib.sha256((text + '|' + language + '|' + voice).encode('utf-8')).hexdigest()}"
    now = time.time()
    cached = TRANSCRIPT_CACHE.get(key)
    if cached and (now - cached["ts"]) < CACHE_TTL_SECONDS:
        mp3_bytes: bytes = cached["payload"]["mp3"]
        return StreamingResponse(
            io.BytesIO(mp3_bytes),
            media_type="audio/mpeg",
            headers={"Content-Disposition": 'inline; filename="speechfindr.mp3"'},
        )

    params = _tts_voice_params(language, voice)
    try:
        tts_kwargs = {
            "text": text,
            "lang": language,
            "slow": bool(params.get("slow", False)),
        }
        tld = params.get("tld")
        # Important: do not pass tld=None to gTTS, otherwise host becomes
        # translate.google.None and request fails.
        if isinstance(tld, str) and tld.strip():
            tts_kwargs["tld"] = tld.strip()
        tts = gTTS(**tts_kwargs)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        mp3_bytes = buf.getvalue()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"TTS generation failed: {exc}") from exc

    TRANSCRIPT_CACHE[key] = {"ts": now, "payload": {"mp3": mp3_bytes}}
    return StreamingResponse(
        io.BytesIO(mp3_bytes),
        media_type="audio/mpeg",
        headers={"Content-Disposition": 'inline; filename="speechfindr.mp3"'},
    )
