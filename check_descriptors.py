"""Check all search_state.json files for Goldschmidt/tau reuse and parent-child similarity."""

import json
import re
import sys
from pathlib import Path


def normalize_code(code: str) -> str:
    """Strip comments, whitespace, and imports for comparison."""
    lines = code.strip().splitlines()
    lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#") and not l.strip().startswith("import")]
    return " ".join(lines)


def is_goldschmidt(code: str) -> bool:
    """Heuristic: check if the code computes (rA + rX) / (sqrt(2) * (rB + rX))."""
    norm = code.replace(" ", "").lower()
    patterns = [
        r"\(ra\s*\+\s*rx\)\s*/\s*\(\s*(?:math\.)?sqrt\(2\)\s*\*\s*\(rb\s*\+\s*rx\)\)",
        r"\(ra\s*\+\s*rx\)\s*/\s*\(\s*1\.414",
        r"\(ra\+rx\)/\(.*sqrt\(2\).*\(rb\+rx\)\)",
        r"ra\+rx\).*1\.4142.*rb\+rx",
    ]
    for p in patterns:
        if re.search(p, norm):
            return True
    # Also check if "goldschmidt" or "tolerance factor" appears in a suspicious way
    if "goldschmidt" in norm or "tolerancefactor" in norm.replace("_", ""):
        return True
    return False


def is_bartel_tau(code: str) -> bool:
    """Heuristic: check if the code computes Bartel's tau."""
    norm = code.replace(" ", "").lower()
    patterns = [
        r"rx\s*/\s*rb.*na.*ln\(ra\s*/\s*rb\)",
        r"rx/rb.*na.*log\(ra/rb\)",
        r"rx/rb.*na.*\(na.*ra/rb.*ln",
    ]
    for p in patterns:
        if re.search(p, norm):
            return True
    return False


def code_similarity(code1: str, code2: str) -> float:
    """Simple token-level Jaccard similarity between two code strings."""
    tokens1 = set(re.findall(r"\w+", normalize_code(code1)))
    tokens2 = set(re.findall(r"\w+", normalize_code(code2)))
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


def analyze_file(path: Path):
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"\n{'='*70}")
        print(f"Run: {path.parent.name}  [SKIPPED — malformed JSON: {e}]")
        print(f"{'='*70}")
        return

    nodes = data["nodes"]
    run_name = path.parent.name
    total_nodes = len(nodes)

    goldschmidt_hits = []
    tau_hits = []
    high_similarity_pairs = []

    for nid, node in nodes.items():
        code = node.get("code", "")
        if is_goldschmidt(code):
            goldschmidt_hits.append(nid)
        if is_bartel_tau(code):
            tau_hits.append(nid)

    for nid, node in nodes.items():
        parent_id = node.get("parent_id")
        if parent_id is None or parent_id not in nodes:
            continue
        parent = nodes[parent_id]
        sim = code_similarity(node["code"], parent["code"])
        if sim > 0.85:
            high_similarity_pairs.append((parent_id, nid, sim))

    print(f"\n{'='*70}")
    print(f"Run: {run_name}  ({total_nodes} nodes)")
    print(f"{'='*70}")

    if goldschmidt_hits:
        print(f"  [WARNING] Goldschmidt tolerance factor detected in: {goldschmidt_hits}")
    else:
        print(f"  [OK] No Goldschmidt tolerance factor found.")

    if tau_hits:
        print(f"  [WARNING] Bartel tau detected in: {tau_hits}")
    else:
        print(f"  [OK] No Bartel tau found.")

    if high_similarity_pairs:
        print(f"  [WARNING] High parent-child similarity (>85% token overlap):")
        for pid, cid, sim in high_similarity_pairs:
            print(f"    parent={pid} -> child={cid}  similarity={sim:.1%}")
    else:
        print(f"  [OK] All children are meaningfully different from parents.")

    # Show a quick summary of parent->child diversity
    parent_child_sims = []
    for nid, node in nodes.items():
        parent_id = node.get("parent_id")
        if parent_id and parent_id in nodes:
            sim = code_similarity(node["code"], nodes[parent_id]["code"])
            parent_child_sims.append(sim)

    if parent_child_sims:
        avg_sim = sum(parent_child_sims) / len(parent_child_sims)
        max_sim = max(parent_child_sims)
        min_sim = min(parent_child_sims)
        print(f"  Parent-child similarity stats: avg={avg_sim:.1%}, min={min_sim:.1%}, max={max_sim:.1%}")


def main():
    search_dir = Path(__file__).parent / "search_runs"
    files = sorted(search_dir.rglob("search_state.json"))

    if not files:
        print("No search_state.json files found.")
        sys.exit(1)

    print(f"Found {len(files)} search run(s).\n")

    for f in files:
        analyze_file(f)

    print()


if __name__ == "__main__":
    main()
