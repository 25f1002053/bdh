from typing import List, Dict, Any

from summarize import _groq_client


LABELS = ["support", "contradict", "irrelevant"]


def label_pairs_llm(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Label pairs using fast heuristic method."""
    out = []
    for p in pairs:
        claim = p["claim_text"]
        chunk = p["chunk_text"]
        label = heuristic_label(claim, chunk)
        out.append({**p, "label": label})
    return out


def heuristic_label(claim: str, chunk: str) -> str:
    import difflib
    ratio = difflib.SequenceMatcher(None, claim.lower(), chunk.lower()).ratio()
    if ratio > 0.4:
        return "support"
    elif ratio < 0.15:
        return "contradict"
    else:
        return "irrelevant"


def aggregate_backstory(claim_labels: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    claim_labels: list for each claim of its labeled top-k pairs.
    Rule: if any claim has majority 'contradict', mark inconsistent. If all claims majority 'support', consistent. Else 'uncertain'.
    """
    backstory_label = "uncertain"
    details = []
    any_contradict = False
    all_support = True
    for pairs in claim_labels:
        counts = {k: 0 for k in LABELS}
        for p in pairs:
            counts[p["label"]] += 1
        details.append(counts)
        if counts["contradict"] > counts["support"]:
            any_contradict = True
        if counts["support"] <= counts["contradict"]:
            all_support = False
    if any_contradict:
        backstory_label = "inconsistent"
    elif all_support:
        backstory_label = "consistent"
    return {"label": backstory_label, "details": details}
