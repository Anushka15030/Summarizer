// Blog Article Summarizer — frontend logic
// Talks to the FastAPI backend's POST /summarize endpoint and renders the
// summary, the compression visualization, and the four stat cards.

const form = document.getElementById("summarize-form");
const urlInput = document.getElementById("url-input");
const submitBtn = document.getElementById("submit-btn");

const loadingEl = document.getElementById("loading");
const loadingTextEl = document.getElementById("loading-text");
const errorEl = document.getElementById("error-banner");
const resultEl = document.getElementById("result");

const summaryTextEl = document.getElementById("summary-text");
const statArticleWords = document.getElementById("stat-article-words");
const statSummaryWords = document.getElementById("stat-summary-words");
const statSentences = document.getElementById("stat-sentences");
const statTime = document.getElementById("stat-time");

const barArticle = document.getElementById("bar-article");
const barSummary = document.getElementById("bar-summary");
const compressionNote = document.getElementById("compression-note");

// Cycles through a couple of status messages while the request is in
// flight, since the backend does a few distinct steps (fetch, extract,
// summarize, validate) that together can take a few seconds.
const LOADING_MESSAGES = [
  "Fetching the article…",
  "Extracting the article body…",
  "Summarizing with Gemini…",
  "Validating word and sentence limits…",
];
let loadingMessageTimer = null;

function startLoadingMessages() {
  let index = 0;
  loadingTextEl.textContent = LOADING_MESSAGES[index];
  loadingMessageTimer = setInterval(() => {
    index = (index + 1) % LOADING_MESSAGES.length;
    loadingTextEl.textContent = LOADING_MESSAGES[index];
  }, 1800);
}

function stopLoadingMessages() {
  clearInterval(loadingMessageTimer);
  loadingMessageTimer = null;
}

function setLoading(isLoading) {
  loadingEl.hidden = !isLoading;
  submitBtn.disabled = isLoading;
  if (isLoading) {
    startLoadingMessages();
  } else {
    stopLoadingMessages();
  }
}

function showError(message) {
  errorEl.textContent = message;
  errorEl.hidden = false;
}

function clearError() {
  errorEl.hidden = true;
  errorEl.textContent = "";
}

function renderResult(data) {
  summaryTextEl.textContent = data.summary;

  statArticleWords.textContent = data.article_word_count.toLocaleString();
  statSummaryWords.textContent = data.summary_word_count.toLocaleString();
  statSentences.textContent = data.summary_sentence_count;
  statTime.textContent = `${data.processing_time}s`;

  // Compression bars: article bar is always the 100% baseline, the summary
  // bar is sized relative to it so the reduction is visible at a glance.
  const articleWords = data.article_word_count || 1; // guard divide-by-zero
  const summaryRatio = Math.min(data.summary_word_count / articleWords, 1);
  const reductionPct = Math.round((1 - summaryRatio) * 100);

  barArticle.style.width = "100%";
  // Defer to next frame so the width transition actually animates from 0.
  requestAnimationFrame(() => {
    barSummary.style.width = `${Math.max(summaryRatio * 100, 2)}%`;
  });

  compressionNote.textContent =
    `${reductionPct}% shorter — ${data.article_word_count.toLocaleString()} → ` +
    `${data.summary_word_count.toLocaleString()} words`;

  resultEl.hidden = false;
}

async function handleSubmit(event) {
  event.preventDefault();

  const url = urlInput.value.trim();
  if (!url) {
    showError("Please enter a blog article URL.");
    return;
  }

  clearError();
  resultEl.hidden = true;
  barSummary.style.width = "0%";
  setLoading(true);

  try {
    const response = await fetch("/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });

    const data = await response.json();

    if (!response.ok) {
      // FastAPI's HTTPException body looks like { "detail": "..." }
      const message =
        (data && data.detail) || `Request failed with status ${response.status}.`;
      showError(message);
      return;
    }

    renderResult(data);
  } catch (err) {
    // Covers network failures (server unreachable, CORS, DNS, offline, etc.)
    showError("Could not reach the server. Check your connection and try again.");
  } finally {
    setLoading(false);
  }
}

form.addEventListener("submit", handleSubmit);
