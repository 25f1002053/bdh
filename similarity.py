from typing import List, Dict, Any, Tuple
import torch


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a / (a.norm() + 1e-8)
    b = b / (b.norm() + 1e-8)
    return float(torch.dot(a, b).item())


def top_k_related(
    claim_emb: torch.Tensor,
    chunk_states: List[Tuple[int, torch.Tensor, torch.Tensor, Dict[str, Any]]],
    k: int = 10,
    require_character: str | None = None,
) -> List[Dict[str, Any]]:
    """
    chunk_states items: (chunk_index, global_state, char_state, chunk_record)
    Filters by character presence if require_character is provided; sorts by similarity to char_state.
    """
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for idx, g_state, c_state, rec in chunk_states:
        if require_character and require_character not in (rec.get("characters") or []):
            continue
        dist = cosine(claim_emb, c_state)
        scored.append((dist, {
            "chunk_index": idx,
            "chunk_text": rec.get("raw_text", ""),
            "characters": rec.get("characters", []),
            "char_state": c_state.cpu().tolist(),
            "global_state": g_state.cpu().tolist(),
            "similarity": dist,
        }))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scored[:k]]
