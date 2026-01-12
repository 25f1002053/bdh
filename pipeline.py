import os
import json
import logging
from typing import List, Dict, Any, Tuple

import torch
from dotenv import load_dotenv

from chunking import make_chunks
from summarize import summarize_chunks
from claims import extract_claims
from bdh_hf import BDHRecurrent
from similarity import top_k_related
from classifiers import label_pairs_llm, aggregate_backstory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def process_novel(path: str, n_sent_per_chunk: int = 6, overlap: int = 2) -> List[Dict[str, Any]]:
    logger.info(f"Processing novel: {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    logger.info(f"  Loaded {len(text)} characters")
    chunks = make_chunks(text, n_sent_per_chunk=n_sent_per_chunk, overlap=overlap)
    logger.info(f"  Created {len(chunks)} chunks with n_sent={n_sent_per_chunk}, overlap={overlap}")
    chunks = summarize_chunks(chunks)
    logger.info(f"  Generated summaries for all {len(chunks)} chunks")
    return chunks


def run_state_tracking(
    chunks: List[Dict[str, Any]],
    target_character: str,
) -> List[Tuple[int, torch.Tensor, torch.Tensor, Dict[str, Any]]]:
    logger.info(f"Running BDH state tracking for character: {target_character}")
    bdh = BDHRecurrent()
    g, c = bdh.init_states()
    history = []
    for i, ch in enumerate(chunks):
        idx = ch["chunk_index"]
        text = ch["raw_text"]
        present = target_character in (ch.get("characters") or [])
        g, c, _ = bdh.step(text=text, prev_global=g, prev_char=c, character_present=present)
        history.append((idx, g.clone(), c.clone(), ch))
        if (i + 1) % max(1, len(chunks) // 5) == 0 or (i + 1) == len(chunks):
            logger.info(f"  Processed {i + 1}/{len(chunks)} chunks (character present: {present})")
    logger.info(f"Completed state tracking: {len(history)} states saved")
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
    logger.info("=" * 80)
    logger.info("Starting BDH Narrative Consistency Pipeline")
    logger.info(f"Target character: {target_character}")
    logger.info(f"Output directory: {out_dir}")
    logger.info("=" * 80)
    
    # 1-3: Load novels, chunk, summarize
    logger.info("\n[STAGE 1-3] Loading novels, chunking, and summarizing...")
    novel_files = [os.path.join(novels_dir, f) for f in os.listdir(novels_dir) if f.endswith(".txt")]
    logger.info(f"Found {len(novel_files)} novel files: {[os.path.basename(f) for f in novel_files]}")
    all_novel_chunks: Dict[str, List[Dict[str, Any]]] = {}
    for nf in novel_files:
        chunks = process_novel(nf, n_sent_per_chunk=n_sent_per_chunk, overlap=overlap)
        all_novel_chunks[os.path.basename(nf)] = chunks
        save_json({str(c["chunk_index"]): {k: c[k] for k in ["raw_text", "characters", "summary"]} for c in chunks}, os.path.join(out_dir, os.path.basename(nf) + ".chunks.json"))
        logger.info(f"✓ Saved chunks for {os.path.basename(nf)}")

    # 6-8: BDH recurrent state tracking per novel
    logger.info("\n[STAGE 6-8] Running BDH recurrent state tracking...")
    state_histories: Dict[str, List[Tuple[int, torch.Tensor, torch.Tensor, Dict[str, Any]]]] = {}
    for name, chunks in all_novel_chunks.items():
        logger.info(f"Processing novel: {name}")
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
        logger.info(f"✓ Saved states for {name} ({len(history)} chunks)")

    # 4-5, 10-13: Claims extraction and pairing
    logger.info("\n[STAGE 4-5, 10-13] Extracting claims and building pairs...")
    claims = extract_claims(backstory_text, default_character=target_character)
    logger.info(f"Extracted {len(claims)} claims from backstory")
    save_json(claims, os.path.join(out_dir, "backstory.claims.json"))
    logger.info(f"✓ Saved claims to backstory.claims.json")
    
    bdh = BDHRecurrent()
    pairs_by_claim: List[List[Dict[str, Any]]] = []
    for i, claim in enumerate(claims):
        logger.info(f"Processing claim {i + 1}/{len(claims)}: {claim['text'][:60]}...")
        # concatenate histories from both novels for ranking
        merged_history = []
        for hist in state_histories.values():
            merged_history.extend(hist)
        pairs = build_pairs_for_claim(claim, bdh, merged_history)
        logger.info(f"  Found {len(pairs)} related chunks (top-10)")
        labeled = label_pairs_llm(pairs)
        logger.info(f"  Labeled pairs: {sum(1 for p in labeled if p['label'] == 'support')} support, {sum(1 for p in labeled if p['label'] == 'contradict')} contradict")
        pairs_by_claim.append(labeled)
    save_json(pairs_by_claim, os.path.join(out_dir, "claim_chunk_pairs.json"))
    logger.info(f"✓ Saved claim-chunk pairs to claim_chunk_pairs.json")

    # 16: Backstory-level classification
    logger.info("\n[STAGE 16] Aggregating to backstory consistency...")
    agg = aggregate_backstory(pairs_by_claim)
    logger.info(f"Backstory label: {agg['label'].upper()}")
    save_json(agg, os.path.join(out_dir, "backstory_consistency.json"))
    logger.info(f"✓ Saved backstory consistency to backstory_consistency.json")

    # 17: Reasoning (reuse summaries / brief LLM rationale)
    logger.info("\n[STAGE 17] Generating reasoning...")
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
    logger.info(f"✓ Saved reasoning to reasoning.json")
    
    logger.info("\n" + "=" * 80)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 80)


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

    try:
        run_pipeline(
            novels_dir=args.novels_dir,
            backstory_text=backstory_text,
            target_character=args.target_character,
            out_dir=args.out_dir,
            n_sent_per_chunk=args.n_sent_per_chunk,
            overlap=args.overlap,
        )
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise
