from typing import Dict, Any, List


def _groq_client():
    try:
        import os
        from groq import Groq
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            return None
        return Groq(api_key=key)
    except Exception:
        return None


def summarize_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate 1-2 line summaries for each chunk using Groq if available, else fallback."""
    client = _groq_client()
    out = []
    for ch in chunks:
        raw = ch["raw_text"]
        if client:
            try:
                prompt = (
                    "Summarize the following novel excerpt in 1-2 concise sentences, "
                    "keeping chronological order and naming main characters if present.\n\n"
                    f"Excerpt:\n{raw}\n\nSummary:"
                )
                resp = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.1-8b-instant",
                    temperature=0.3,
                    max_tokens=64,
                )
                summary = resp.choices[0].message.content.strip()
            except Exception:
                summary = _fallback_summary(raw)
        else:
            summary = _fallback_summary(raw)
        out.append({**ch, "summary": summary})
    return out


def _fallback_summary(text: str) -> str:
    # Very lightweight fallback: take first sentence and compress length
    s = text.strip().split(".")
    first = (s[0] if s else text).strip()
    return (first[:200] + ("..." if len(first) > 200 else ""))
