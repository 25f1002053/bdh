import os
import json
from typing import List, Dict, Any, Tuple

import torch
from dotenv import load_dotenv

from chunking import make_chunks
from summarize import summarize_chunks
from claims import extract_claims
from bdh_hf import BDHRecurrent
from similarity import top_k_related
from classifiers import label_pairs_llm, aggregate_backstory


def process_novel(path: str, n_sent_per_chunk: int = 6, overlap: int = 2) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    chunks = make_chunks(text, n_sent_per_chunk=n_sent_per_chunk, overlap=overlap)
    chunks = summarize_chunks(chunks)
    return chunks


def run_state_tracking(
    chunks: List[Dict[str, Any]],
    target_character: str,
) -> List[Tuple[int, torch.Tensor, torch.Tensor, Dict[str, Any]]]:
    bdh = BDHRecurrent()
    g, c = bdh.init_states()
    history = []
    for ch in chunks:
        idx = ch["chunk_index"]
        text = ch["raw_text"]
        present = target_character in (ch.get("characters") or [])
        g, c, _ = bdh.step(text=text, prev_global=g, prev_char=c, character_present=present)
        history.append((idx, g.clone(), c.clone(), ch))
    return history


def build_pairs_for_claim(
    claim: Dict[str, Any],
    bdh: BDHRecurrent,
    history: List[Tuple[int, torch.Tensor, torch.Tensor, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    emb = bdh.embed_claim(claim["text"])
    top = top_k_related(emb, history, k=10, require_character=claim.get("character"))
    pairs = []
    for t in top:
        pairs.append({
            "claim_id": claim["id"],
            "claim_text": claim["text"],
            "claim_type": claim["type"],
            "chunk_index": t["chunk_index"],
            "chunk_text": t["chunk_text"],
            "bdh_char_state": t["char_state"],
            "bdh_global_state": t["global_state"],
            "distance": 1.0 - t["similarity"],
        })
    return pairs


def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_pipeline(
    novels_dir: str,
    backstory_text: str,
    target_character: str,
    out_dir: str = "outputs",
    n_sent_per_chunk: int = 6,
    overlap: int = 2,
):
    # Load environment (e.g., GROQ_API_KEY from .env)
    load_dotenv()
    # 1-3: Load novels, chunk, summarize
    novel_files = [os.path.join(novels_dir, f) for f in os.listdir(novels_dir) if f.endswith(".txt")]
    all_novel_chunks: Dict[str, List[Dict[str, Any]]] = {}
    for nf in novel_files:
        chunks = process_novel(nf, n_sent_per_chunk=n_sent_per_chunk, overlap=overlap)
        all_novel_chunks[os.path.basename(nf)] = chunks
        save_json({str(c["chunk_index"]): {k: c[k] for k in ["raw_text", "characters", "summary"]} for c in chunks}, os.path.join(out_dir, os.path.basename(nf) + ".chunks.json"))

    # 6-8: BDH recurrent state tracking per novel
    state_histories: Dict[str, List[Tuple[int, torch.Tensor, torch.Tensor, Dict[str, Any]]]] = {}
    for name, chunks in all_novel_chunks.items():
        history = run_state_tracking(chunks, target_character)
        # Persist states per chunk
        serial = [
            {
                "chunk_index": idx,
                "char_state": c.cpu().tolist(),
                "global_state": g.cpu().tolist(),
                "chunk": rec,
            }
            for (idx, g, c, rec) in history
        ]
        save_json(serial, os.path.join(out_dir, name + ".states.json"))
        state_histories[name] = history

    # 4-5, 10-13: Claims extraction and pairing
    claims = extract_claims(backstory_text, default_character=target_character)
    save_json(claims, os.path.join(out_dir, "backstory.claims.json"))
    bdh = BDHRecurrent()
    pairs_by_claim: List[List[Dict[str, Any]]] = []
    for claim in claims:
        # concatenate histories from both novels for ranking
        merged_history = []
        for hist in state_histories.values():
            merged_history.extend(hist)
        pairs = build_pairs_for_claim(claim, bdh, merged_history)
        labeled = label_pairs_llm(pairs)
        pairs_by_claim.append(labeled)
    save_json(pairs_by_claim, os.path.join(out_dir, "claim_chunk_pairs.json"))

    # 16: Backstory-level classification
    agg = aggregate_backstory(pairs_by_claim)
    save_json(agg, os.path.join(out_dir, "backstory_consistency.json"))

    # 17: Reasoning (reuse summaries / brief LLM rationale)
    # For brevity, attach a one-line rationale using summaries
    rationales = []
    for labeled_pairs in pairs_by_claim:
        for p in labeled_pairs:
            rationales.append({
                "claim_id": p["claim_id"],
                "claim_text": p["claim_text"],
                "chunk_index": p["chunk_index"],
                "label": p["label"],
                "reason": f"Label '{p['label']}' based on semantic proximity and character presence.",
            })
    save_json(rationales, os.path.join(out_dir, "reasoning.json"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BDH pipeline for novels and backstory consistency")
    parser.add_argument("--novels_dir", default="novels")
    parser.add_argument("--target_character", required=True)
    parser.add_argument("--backstory_file", required=True)
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--n_sent_per_chunk", type=int, default=6)
    parser.add_argument("--overlap", type=int, default=2)
    args = parser.parse_args()

    with open(args.backstory_file, "r", encoding="utf-8", errors="ignore") as f:
        backstory_text = f.read()

    run_pipeline(
        novels_dir=args.novels_dir,
        backstory_text=backstory_text,
        target_character=args.target_character,
        out_dir=args.out_dir,
        n_sent_per_chunk=args.n_sent_per_chunk,
        overlap=args.overlap,
    )
