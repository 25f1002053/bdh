from typing import List, Dict, Any
import uuid

from summarize import _groq_client


CLAIM_TYPES = ["event", "trait", "world_rule"]
TIME_BUCKETS = ["childhood", "adulthood", "unspecified"]
STABILITY_BUCKETS = ["short_term", "long_term", "unspecified"]


def extract_claims(backstory_text: str, default_character: str = "Unknown") -> List[Dict[str, Any]]:
    client = _groq_client()
    if client:
        try:
            prompt = (
                "Extract claims from the backstory. Return JSON array; each item must "
                "have keys: id, character, type (event|trait|world_rule), time_scope "
                "(childhood|adulthood|unspecified), stability (short_term|long_term|unspecified), text. "
                "Be precise and split multi-fact sentences into multiple claims.\n\nBackstory:\n"
                f"{backstory_text}\n\nJSON:"
            )
            resp = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.2,
                max_tokens=512,
            )
            content = resp.choices[0].message.content.strip()
            import json
            data = json.loads(content)
            # Normalize and validate
            return [normalize_claim(c, default_character) for c in data]
        except Exception:
            return heuristic_claims(backstory_text, default_character)
    else:
        return heuristic_claims(backstory_text, default_character)


def normalize_claim(c: Dict[str, Any], default_character: str) -> Dict[str, Any]:
    cid = c.get("id") or str(uuid.uuid4())
    character = c.get("character") or default_character
    ctype = c.get("type") if c.get("type") in CLAIM_TYPES else "trait"
    time_scope = c.get("time_scope") if c.get("time_scope") in TIME_BUCKETS else "unspecified"
    stability = c.get("stability") if c.get("stability") in STABILITY_BUCKETS else "unspecified"
    text = c.get("text") or ""
    return {
        "id": cid,
        "character": character,
        "type": ctype,
        "time_scope": time_scope,
        "stability": stability,
        "text": text,
    }


def heuristic_claims(backstory_text: str, default_character: str) -> List[Dict[str, Any]]:
    import re
    sents = re.split(r"(?<=[.!?])\s+", backstory_text.strip())
    claims: List[Dict[str, Any]] = []
    for s in sents:
        if not s.strip():
            continue
        ctype = "event" if any(k in s.lower() for k in ["happened", "went", "met", "died", "married", "left"]) else "trait"
        time_scope = "childhood" if any(k in s.lower() for k in ["child", "young"]) else "adulthood"
        claims.append({
            "id": str(uuid.uuid4()),
            "character": default_character,
            "type": ctype,
            "time_scope": time_scope,
            "stability": "unspecified",
            "text": s.strip(),
        })
    return claims
