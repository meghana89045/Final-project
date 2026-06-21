"""
Core engine for the AI Resume Ranker.
- Extracts text from PDF resumes
- Cleans/preprocesses text
- Scores resumes against a job description using TF-IDF + cosine similarity
- Extracts matched / missing skills for an explainable breakdown
"""
import re
import io

import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOP_WORDS = set(stopwords.words("english"))


# A reasonably broad curated skills list used purely for the "matched / missing
# skills" explainability panel. This does NOT affect the TF-IDF score itself.
SKILL_BANK = [
    "python", "java", "c++", "c", "javascript", "typescript", "sql", "nosql",
    "html", "css", "react", "angular", "vue", "node", "django", "flask",
    "fastapi", "spring", "machine learning", "deep learning", "nlp",
    "natural language processing", "computer vision", "data analysis",
    "data visualization", "pandas", "numpy", "scikit-learn", "sklearn",
    "tensorflow", "keras", "pytorch", "matplotlib", "seaborn", "power bi",
    "tableau", "excel", "aws", "azure", "gcp", "docker", "kubernetes",
    "git", "github", "ci/cd", "linux", "rest api", "api", "mongodb",
    "mysql", "postgresql", "spark", "hadoop", "agile", "scrum",
    "project management", "communication", "leadership", "problem solving",
    "data structures", "algorithms", "oop", "testing", "devops",
    "data mining", "statistics", "r programming", "etl", "big data",
    "android", "ios", "flutter", "swift", "kotlin", "php", "ruby",
    "go", "rust", "blockchain", "cybersecurity", "networking",
]


def extract_text_from_pdf(file_stream) -> str:
    """Extract raw text from a PDF given a file-like object or bytes."""
    if isinstance(file_stream, bytes):
        doc = fitz.open(stream=file_stream, filetype="pdf")
    else:
        data = file_stream.read()
        doc = fitz.open(stream=data, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def preprocess_text(text: str) -> str:
    """Lowercase, strip punctuation, remove stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s+#./]", " ", text)  # keep + # . / for skills like c++, c#, node.js
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


def extract_skills(text: str) -> set:
    """Find which curated skills appear in a block of text."""
    text_lower = text.lower()
    found = set()
    for skill in SKILL_BANK:
        # word-boundary-ish match, tolerant of skills containing symbols
        pattern = re.escape(skill)
        if re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", text_lower):
            found.add(skill)
    return found


def rank_resumes(job_description: str, resumes: list[dict]) -> list[dict]:
    """
    resumes: list of dicts like {"filename": str, "raw_text": str}
    returns: list of dicts with score, matched_skills, missing_skills, sorted desc by score
    """
    jd_clean = preprocess_text(job_description)
    jd_skills = extract_skills(job_description)

    corpus = [jd_clean] + [preprocess_text(r["raw_text"]) for r in resumes]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    jd_vec = tfidf_matrix[0:1]
    resume_vecs = tfidf_matrix[1:]

    scores = cosine_similarity(jd_vec, resume_vecs)[0]

    results = []
    for i, r in enumerate(resumes):
        resume_skills = extract_skills(r["raw_text"])
        matched = sorted(jd_skills & resume_skills)
        missing = sorted(jd_skills - resume_skills)
        results.append({
            "filename": r["filename"],
            "score": round(float(scores[i]) * 100, 2),
            "matched_skills": matched,
            "missing_skills": missing,
            "snippet": r["raw_text"].strip().split("\n")[0][:120] if r["raw_text"].strip() else "No preview available",
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results
