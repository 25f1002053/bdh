import csv
from typing import List, Dict, Any


def read_csv(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    return rows


def to_backstory(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row.get("id"),
        "book_name": row.get("book_name"),
        "character": row.get("char"),
        "caption": row.get("caption"),
        "text": row.get("content") or "",
        "label": row.get("label"),
    }
