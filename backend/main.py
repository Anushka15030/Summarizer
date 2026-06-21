"""
Blog Article Summarizer — FastAPI backend
==========================================

This is the original CLI summarizer (fetch -> extract -> summarize -> validate)
wrapped behind a single HTTP endpoint, with the frontend served from the same
process so the whole app deploys as one Render web service.

The pipeline functions below (validate_url, fetch_html, extract_main_text,
build_prompt, summarize_with_gemini, split_into_sentences, enforce_limits,
save_summary) are unchanged from the original CLI version. Only the
orchestration layer at the bottom of this file is new.

Environment variables (see .env.example):
    GEMINI_API_KEY   Required. Your Gemini API key.
    GEMINI_MODEL     Optional. Defaults to "gemini-2.5-flash".
"""

import os
import re
import time
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

MAX_WORDS = 300
MAX_SENTENCES = 20
FETCH_TIMEOUT_SECONDS = 15
API_TIMEOUT_SECONDS = 60
MIN_ARTICLE_CHARS = 200  # below this, we treat the page as having no real article
DEFAULT_MODEL = "gemini-2.5-flash"
USER_AGENT = "Mozilla/5.0 (compatible; BlogSummarizerBot/1.0)"

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class SummarizerError(Exception):
    """Raised for any expected failure, so the API layer can report it cleanly."""


# --------------------------------------------------------------------------
# Step 1: Fetch and extract the article  (unchanged from the CLI version)
# --------------------------------------------------------------------------

def validate_url(url: str) -> None:
    """Raise SummarizerError if `url` is not a well-formed http/https URL."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise SummarizerError(f"'{url}' is not a valid http/https URL.")


def fetch_html(url: str) -> str:
    """Download the raw HTML for `url`, raising SummarizerError on any failure."""
    try:
        response = requests.get(
            url, timeout=FETCH_TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT}
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise SummarizerError(f"Timed out while fetching {url}.")
    except requests.exceptions.ConnectionError:
        raise SummarizerError(f"Network error while trying to reach {url}.")
    except requests.exceptions.HTTPError as exc:
        raise SummarizerError(f"Server returned an error for {url}: {exc}")
    except requests.exceptions.RequestException as exc:
        raise SummarizerError(f"Failed to fetch {url}: {exc}")
    return response.text


def extract_main_text(html: str) -> str:
    """
    Pull the readable article body out of a raw HTML page.

    Strategy: strip non-content tags (scripts, nav, footers, etc.), then
    prefer an <article> or <main> element if the page has one, and fall back
    to every <p> tag on the page otherwise. This covers the vast majority of
    blog layouts without needing site-specific scraping rules.
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "header", "footer", "noscript", "form", "svg"]):
        tag.decompose()

    container = soup.find("article") or soup.find("main")
    paragraphs = container.find_all("p") if container else soup.find_all("p")

    text = "\n".join(p.get_text(" ", strip=True) for p in paragraphs)
    text = re.sub(r"\n{2,}", "\n", text).strip()

    if len(text) < MIN_ARTICLE_CHARS:
        raise SummarizerError(
            "Could not find a substantial article body on this page. "
            "It may use an unsupported layout, require JavaScript to render, "
            "or simply have no meaningful content."
        )
    return text


# --------------------------------------------------------------------------
# Step 2: Summarize with Gemini  (unchanged from the CLI version)
# --------------------------------------------------------------------------

def build_prompt(article_text: str) -> str:
    """Build the instruction sent to Gemini. Limits are stated but never trusted blindly."""
    return (
        "Summarize the following blog article in plain prose.\n"
        f"Hard requirements: no more than {MAX_WORDS} words, and no more than "
        f"{MAX_SENTENCES} sentences. Do not use bullet points, headings, or markdown "
        "formatting. Write clear, well-formed sentences that capture the key facts "
        "and takeaways.\n\n"
        f"Article:\n{article_text}"
    )


def summarize_with_gemini(article_text: str, api_key: str, model: str) -> str:
    """Call the Gemini API and return the raw summary text it generated."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {"contents": [{"parts": [{"text": build_prompt(article_text)}]}]}

    try:
        response = requests.post(url, json=payload, timeout=API_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise SummarizerError("Timed out while waiting for the Gemini API.")
    except requests.exceptions.HTTPError as exc:
        raise SummarizerError(f"Gemini API returned an error: {exc}")
    except requests.exceptions.RequestException as exc:
        raise SummarizerError(f"Failed to reach the Gemini API: {exc}")

    data = response.json()
    try:
        summary = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        reason = (data.get("candidates") or [{}])[0].get("finishReason", "unknown")
        raise SummarizerError(
            f"Gemini API returned no usable summary (finishReason={reason})."
        )

    if not summary.strip():
        raise SummarizerError("Gemini API returned an empty summary.")

    return summary.strip()


# --------------------------------------------------------------------------
# Step 3: Validate and enforce the word/sentence limits  (unchanged)
# --------------------------------------------------------------------------

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using terminal punctuation as the boundary."""
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []
    pieces = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', text)
    return [p.strip() for p in pieces if p.strip()]


def enforce_limits(
    summary: str, max_words: int = MAX_WORDS, max_sentences: int = MAX_SENTENCES
) -> str:
    """
    Trim `summary` at sentence boundaries until it satisfies both the word
    and sentence limits. A sentence is only included if adding it keeps the
    summary within both budgets, so we never cut a sentence in half.
    """
    sentences = split_into_sentences(summary)
    if not sentences:
        return ""

    kept: List[str] = []
    word_count = 0
    for sentence in sentences:
        if len(kept) >= max_sentences:
            break
        sentence_words = len(sentence.split())
        if word_count + sentence_words > max_words:
            break
        kept.append(sentence)
        word_count += sentence_words

    if not kept:
        # Edge case: even the first sentence alone exceeds the word budget.
        # Hard-trim it by word count rather than returning nothing.
        kept = [" ".join(sentences[0].split()[:max_words])]

    return " ".join(kept)


# --------------------------------------------------------------------------
# Step 4: Save output  (unchanged in behavior; path is now a fixed directory)
# --------------------------------------------------------------------------

def save_summary(summary: str, path: Path) -> None:
    """Write the final summary to a text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# API layer (new): request/response models + the /summarize endpoint
# --------------------------------------------------------------------------

class SummarizeRequest(BaseModel):
    url: str = Field(..., description="URL of the blog article to summarize")


class SummarizeResponse(BaseModel):
    summary: str
    article_word_count: int
    summary_word_count: int
    summary_sentence_count: int
    processing_time: float


load_dotenv()

app = FastAPI(title="Blog Article Summarizer API")

# Allows the frontend to call the API even when served from a different
# origin (e.g. running the frontend with a separate local dev server).
# Same-origin requests (the normal deployed setup, see StaticFiles mount
# below) work regardless of this setting.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest) -> SummarizeResponse:
    """Fetch, extract, summarize, and validate a blog article. Returns JSON stats."""
    start_time = time.time()

    api_key = os.environ.get("GEMINI_API_KEY")
    model = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
    if not api_key:
        raise HTTPException(
            status_code=500, detail="GEMINI_API_KEY is not configured on the server."
        )

    try:
        validate_url(request.url)
        html = fetch_html(request.url)
        article_text = extract_main_text(html)
        raw_summary = summarize_with_gemini(article_text, api_key, model)
        summary = enforce_limits(raw_summary)
    except SummarizerError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    save_summary(summary, OUTPUT_DIR / "summary.txt")

    return SummarizeResponse(
        summary=summary,
        article_word_count=len(article_text.split()),
        summary_word_count=len(summary.split()),
        summary_sentence_count=len(split_into_sentences(summary)),
        processing_time=round(time.time() - start_time, 2),
    )


@app.get("/health")
def health() -> dict:
    """Simple liveness check, useful for Render's health checks."""
    return {"status": "ok"}


# Serve the frontend (index.html, style.css, script.js) from the same
# process. Mounted last so it doesn't shadow the API routes above.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
