"""Retrieval logic for CampusBuddy AI (no Streamlit, so it can be unit tested)."""

import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = Path(__file__).resolve().parent / "data" / "faqs.csv"
REQUIRED_COLUMNS = ["category", "question", "answer"]

# Below this cosine similarity we say "no close match" instead of guessing.
MIN_SCORE = 0.25

# Tiny alias table: common abbreviations/synonyms students actually type.
# Whole-word, case-insensitive; the expansion is appended to the query.
ALIASES = {
    "ai": "artificial intelligence",
    "ml": "machine learning",
    "nlp": "natural language processing",
    "dict": "dictionary",
    "finals": "exams",
    "exam": "exams",
    "test": "exams",
    "cv": "portfolio",
    "resume": "portfolio",
    "procrastinate": "procrastination",
    "procrastinating": "procrastination",
    "timetable": "study timetable schedule",
    "schedule": "timetable",
    "github": "git hub github",
}


def load_faqs(path=DATA_FILE):
    """Load the FAQ CSV and validate it. Raises ValueError on a bad file."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"FAQ file not found: {path.name}")
    faqs = pd.read_csv(path).fillna("")
    missing = [c for c in REQUIRED_COLUMNS if c not in faqs.columns]
    if missing:
        raise ValueError(f"FAQ file is missing column(s): {', '.join(missing)}")
    for column in REQUIRED_COLUMNS:
        faqs[column] = faqs[column].astype(str).str.strip()
    faqs = faqs[(faqs["question"] != "") & (faqs["answer"] != "")]
    faqs = faqs.drop_duplicates(subset="question").reset_index(drop=True)
    if faqs.empty:
        raise ValueError("FAQ file has no usable rows.")
    return faqs


def expand_query(text):
    """Append alias expansions for abbreviations found in the query."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    extra = [ALIASES[w] for w in words if w in ALIASES]
    return text if not extra else f"{text} {' '.join(extra)}"


class FaqSearcher:
    """TF-IDF + cosine similarity over question (weighted) and answer text."""

    def __init__(self, faqs):
        self.faqs = faqs.reset_index(drop=True)
        # The question is repeated so it counts more than the answer text.
        documents = (self.faqs["question"] + " ") * 2 + self.faqs["answer"]
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2)
        )
        self.matrix = self.vectorizer.fit_transform(documents)

    def search(self, user_question, top_k=1):
        """Return a list of (row, score) pairs, best first."""
        vector = self.vectorizer.transform([expand_query(user_question)])
        scores = cosine_similarity(vector, self.matrix).ravel()
        order = scores.argsort()[::-1][:top_k]
        return [(self.faqs.iloc[i], float(scores[i])) for i in order]

    def best_match(self, user_question):
        """Return (row, score), or (None, score) if below MIN_SCORE."""
        row, score = self.search(user_question, top_k=1)[0]
        return (row, score) if score >= MIN_SCORE else (None, score)
