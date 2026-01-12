import os
import json
import pickle
from typing import List, Dict, Any

import numpy as np
from sklearn.linear_model import LogisticRegression


def load_pairs(path: str) -> List[List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def train_pair_classifier(pairs_by_claim: List[List[Dict[str, Any]]], out_path: str):
    X = []
    y = []
    labmap = {"support": 0, "irrelevant": 1, "contradict": 2}
    for cl in pairs_by_claim:
        for p in cl:
            X.append([p.get("distance", 0.0)])
            y.append(labmap.get(p.get("label", "irrelevant"), 1))
    X = np.array(X)
    y = np.array(y)
    if len(set(y)) < 2:
        print("Not enough label diversity to train; skipping.")
        return
    clf = LogisticRegression(max_iter=1000, multi_class="auto")
    clf.fit(X, y)
    with open(out_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Saved pair classifier to {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train simple classifiers on claim-chunk pairs")
    parser.add_argument("--pairs_file", default="outputs/claim_chunk_pairs.json")
    parser.add_argument("--out_path", default="outputs/pair_classifier.pkl")
    args = parser.parse_args()
    data = load_pairs(args.pairs_file)
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    train_pair_classifier(data, args.out_path)


if __name__ == "__main__":
    main()
