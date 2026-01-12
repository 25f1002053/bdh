import os
import json
import logging
from typing import List, Dict, Any

from dotenv import load_dotenv

from dataset import read_csv, to_backstory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
from pipeline import process_novel, run_state_tracking, build_pairs_for_claim, save_json
from claims import extract_claims
from bdh_hf import BDHRecurrent
from classifiers import label_pairs_llm, aggregate_backstory


def features_from_pairs(pairs: List[Dict[str, Any]]) -> Dict[str, float]:
    support = sum(1 for p in pairs if p["label"] == "support")
    contradict = sum(1 for p in pairs if p["label"] == "contradict")
    irrelevant = sum(1 for p in pairs if p["label"] == "irrelevant")
    total = max(1, len(pairs))
    mean_distance = sum(p.get("distance", 0.0) for p in pairs) / total
    return {
        "frac_support": support / total,
        "frac_contradict": contradict / total,
        "frac_irrelevant": irrelevant / total,
        "mean_distance": mean_distance,
    }


def ensure_states(novels_dir: str, target_character: str, out_dir: str) -> Dict[str, Any]:
    # Precompute chunks and states per novel for the target character
    novel_files = [os.path.join(novels_dir, f) for f in os.listdir(novels_dir) if f.endswith(".txt")]
    state_histories = {}
    for nf in novel_files:
        name = os.path.basename(nf)
        chunks_path = os.path.join(out_dir, name + ".chunks.json")
        states_path = os.path.join(out_dir, name + ".states.json")
        if os.path.exists(chunks_path) and os.path.exists(states_path):
            with open(states_path, "r", encoding="utf-8") as f:
                import json
                serial = json.load(f)
            # reconstruct minimal history tuples
            hist = []
            for item in serial:
                import torch
                idx = item["chunk_index"]
                g = torch.tensor(item["global_state"])  # (D)
                c = torch.tensor(item["char_state"])    # (D)
                rec = item["chunk"]
                hist.append((idx, g, c, rec))
            state_histories[name] = hist
            continue
        chunks = process_novel(nf)
        history = run_state_tracking(chunks, target_character)
        # save as in pipeline
        serial = [
            {
                "chunk_index": idx,
                "char_state": c.cpu().tolist(),
                "global_state": g.cpu().tolist(),
                "chunk": rec,
            }
            for (idx, g, c, rec) in history
        ]
        save_json(serial, states_path)
        state_histories[name] = history
    return state_histories


def run_train_test(train_csv: str, test_csv: str, novels_dir: str, out_dir: str = "outputs"):
    load_dotenv()
    os.makedirs(out_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Starting BDH Train/Test Pipeline")
    logger.info(f"Train CSV: {train_csv}, Test CSV: {test_csv}")
    logger.info(f"Novels dir: {novels_dir}, Output dir: {out_dir}")
    logger.info("=" * 80)
    
    logger.info("\n[LOAD] Reading CSVs...")
    train_rows = [to_backstory(r) for r in read_csv(train_csv)]
    test_rows = [to_backstory(r) for r in read_csv(test_csv)]
    logger.info(f"✓ Loaded {len(train_rows)} train rows and {len(test_rows)} test rows")

    # Precompute histories per character across novels
    logger.info("\n[PRECOMPUTE] Building character histories across novels...")
    unique_chars = sorted(set([r["character"] for r in train_rows + test_rows if r["character"]]))
    logger.info(f"Found {len(unique_chars)} unique characters: {unique_chars}")

    histories_by_char = {}
    for char in unique_chars:
        logger.info(f"  Processing character: {char}")
        histories_by_char[char] = ensure_states(novels_dir, char, out_dir)
        logger.info(f"  ✓ Histories precomputed for {char}")

    # Build features for train
    logger.info("\n[TRAIN] Building features and training classifier...")
    bdh = BDHRecurrent()
    train_features = []
    train_labels = []
    train_evidence = []
    
    for i, r in enumerate(train_rows):
        char = r["character"]
        logger.info(f"  Processing train row {i + 1}/{len(train_rows)}: {r['id']} ({char})")
        claims = extract_claims(r["text"], default_character=char)
        logger.info(f"    Extracted {len(claims)} claims")
        # merge histories across novels
        merged_history = []
        for hist in histories_by_char[char].values():
            merged_history.extend(hist)
        pairs_by_claim = []
        for j, claim in enumerate(claims):
            pairs = label_pairs_llm(build_pairs_for_claim(claim, bdh, merged_history))
            logger.info(f"      Claim {j + 1}/{len(claims)}: {len(pairs)} labeled pairs")
            pairs_by_claim.append(pairs)
        agg = aggregate_backstory(pairs_by_claim)
        logger.info(f"    Backstory label: {agg['label']}")
        feats = features_from_pairs([p for plist in pairs_by_claim for p in plist])
        train_features.append([feats["frac_support"], feats["frac_contradict"], feats["mean_distance"]])
        train_labels.append(1 if (agg["label"] == "consistent") else 0)
        train_evidence.append({"id": r["id"], "pairs": pairs_by_claim, "agg": agg})
    
    logger.info(f"✓ Built {len(train_features)} train samples")
    save_json(train_evidence, os.path.join(out_dir, "train_evidence.json"))
    logger.info(f"✓ Saved train evidence to train_evidence.json")

    # Train a simple classifier
    logger.info("\n[CLASSIFIER] Training logistic regression on train features...")
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    X = np.array(train_features)
    y = np.array(train_labels)
    logger.info(f"  Feature matrix shape: {X.shape}, Labels: {np.bincount(y)}")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    logger.info(f"✓ Classifier trained with coefficients: {clf.coef_[0]}")
    
    # Save classifier
    import pickle
    with open(os.path.join(out_dir, "backstory_classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)
    logger.info(f"✓ Saved classifier to backstory_classifier.pkl")

    # Predict on test
    logger.info("\n[TEST] Predicting on test set...")
    test_preds = []
    test_evidence = []
    for i, r in enumerate(test_rows):
        char = r["character"]
        logger.info(f"  Processing test row {i + 1}/{len(test_rows)}: {r['id']} ({char})")
        claims = extract_claims(r["text"], default_character=char)
        logger.info(f"    Extracted {len(claims)} claims")
        merged_history = []
        for hist in histories_by_char[char].values():
            merged_history.extend(hist)
        pairs_by_claim = []
        for j, claim in enumerate(claims):
            pairs = label_pairs_llm(build_pairs_for_claim(claim, bdh, merged_history))
            logger.info(f"      Claim {j + 1}/{len(claims)}: {len(pairs)} labeled pairs")
            pairs_by_claim.append(pairs)
        agg = aggregate_backstory(pairs_by_claim)
        logger.info(f"    Backstory label: {agg['label']}")
        feats = features_from_pairs([p for plist in pairs_by_claim for p in plist])
        proba = clf.predict_proba([[feats["frac_support"], feats["frac_contradict"], feats["mean_distance"]]])[0]
        pred_label = "consistent" if proba[1] >= 0.5 else "inconsistent"
        logger.info(f"    Predicted: {pred_label} (confidence: {max(proba):.3f})")
        test_preds.append({"id": r["id"], "label": pred_label})
        test_evidence.append({"id": r["id"], "pairs": pairs_by_claim, "agg": agg, "features": feats})
    
    logger.info(f"✓ Generated {len(test_preds)} test predictions")
    save_json(test_evidence, os.path.join(out_dir, "test_evidence.json"))
    logger.info(f"✓ Saved test evidence to test_evidence.json")
    
    # Write predictions CSV
    logger.info("\n[OUTPUT] Writing predictions CSV...")
    with open(os.path.join(out_dir, "test_predictions.csv"), "w", encoding="utf-8") as f:
        f.write("id,label\n")
        for p in test_preds:
            f.write(f"{p['id']},{p['label']}\n")
    logger.info(f"✓ Saved predictions to test_predictions.csv")
    
    logger.info("\n" + "=" * 80)
    logger.info("Train/Test pipeline completed successfully!")
    logger.info(f"Predictions: {os.path.join(out_dir, 'test_predictions.csv')}")
    logger.info("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run train/test using BDH pipeline features and a simple classifier")
    parser.add_argument("--train_csv", default="train.csv")
    parser.add_argument("--test_csv", default="test.csv")
    parser.add_argument("--novels_dir", default="novels")
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()
    
    try:
        run_train_test(args.train_csv, args.test_csv, args.novels_dir, args.out_dir)
    except Exception as e:
        logger.error(f"Dataset pipeline failed with error: {e}", exc_info=True)
        raise
