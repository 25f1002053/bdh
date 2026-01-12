import re
from typing import List, Dict, Any


def split_sentences(text: str) -> List[str]:
    # Simple sentence splitter; avoids heavy dependencies
    # Splits on ., !, ? while preserving basic punctuation
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def extract_characters(text: str) -> List[str]:
    """Heuristic character extraction: find capitalized name-like tokens.
    Falls back to regex; uses spaCy PERSON if available.
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        names = sorted(set([ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]))
        if names:
            return names
    except Exception:
        pass
    # Fallback: sequences of capitalized words (2+ tokens)
    candidates = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text)
    uniq = sorted(set([c.strip() for c in candidates]))
    return uniq


def make_chunks(text: str, n_sent_per_chunk: int = 6, overlap: int = 2) -> List[Dict[str, Any]]:
    sentences = split_sentences(text)
    chunks: List[Dict[str, Any]] = []
    if not sentences:
        return chunks
    step = max(1, n_sent_per_chunk - overlap)
    idx = 0
    chunk_index = 0
    while idx < len(sentences):
        sel = sentences[idx : idx + n_sent_per_chunk]
        raw = " ".join(sel).strip()
        chars = extract_characters(raw)
        chunks.append({
            "chunk_index": chunk_index,
            "raw_text": raw,
            "characters": chars,
        })
        chunk_index += 1
        if idx + n_sent_per_chunk >= len(sentences):
            break
        idx += step
    return chunks
