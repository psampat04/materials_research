"""Before/after comparison: single hold-out split vs 5-fold cross-validation.

Usage:
    uv run python compare_cv.py
    uv run python compare_cv.py search_runs/20260314_160923/search_state.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evaluator import evaluate_candidate, load_dataset

DATA_PATH = "perovskite-stability/TableS1.csv"
DECISION_TREE_MAX_DEPTH = 2
TRAIN_SPLIT_LABEL = 1
CV_FOLDS = 5


def load_nodes(state_path: Path) -> list[dict]:
    with open(state_path) as f:
        data = json.load(f)
    return list(data["nodes"].values())


def pick_representative_nodes(nodes: list[dict], n: int = 8) -> list[dict]:
    """Pick a spread of nodes by test_accuracy for a meaningful comparison."""
    valid = [nd for nd in nodes if nd.get("code") and not nd.get("error")]
    valid.sort(key=lambda nd: nd.get("metrics", {}).get("test_accuracy", 0), reverse=True)
    # Take top, bottom, and middle
    if len(valid) <= n:
        return valid
    step = max(1, len(valid) // n)
    return [valid[i] for i in range(0, min(len(valid), step * n), step)][:n]


def run_comparison(state_path: Path, df: pd.DataFrame, tmp_dir: Path) -> pd.DataFrame:
    nodes = load_nodes(state_path)
    sample = pick_representative_nodes(nodes)

    rows = []
    for nd in sample:
        node_id = nd["id"]
        code = nd.get("code", "")
        if not code:
            continue

        # --- BEFORE: single hold-out split (no CV, cv_folds=0 bypasses CV) ---
        before = evaluate_candidate(
            code, df, tmp_dir, node_id + "_before",
            decision_tree_max_depth=DECISION_TREE_MAX_DEPTH,
            train_split_label=TRAIN_SPLIT_LABEL,
            cv_folds=0,  # disabled — use hold-out only
        )

        # --- AFTER: 5-fold cross-validation ---
        after = evaluate_candidate(
            code, df, tmp_dir, node_id + "_after",
            decision_tree_max_depth=DECISION_TREE_MAX_DEPTH,
            train_split_label=TRAIN_SPLIT_LABEL,
            cv_folds=CV_FOLDS,
        )

        if before.error or after.error:
            continue

        rows.append({
            "node_id": node_id,
            "formula": (nd.get("formula") or "")[:60],
            # --- BEFORE (hold-out) ---
            "before_train_acc": before.train_accuracy,
            "before_test_acc": before.test_accuracy,
            # The old accuracy was the full-dataset prediction accuracy
            "before_overall_acc": before.test_accuracy,  # same as test for hold-out
            # --- AFTER (5-fold CV) ---
            "after_cv_mean": after.cv_accuracy,
            "after_cv_std": after.cv_std,
            "after_cv_folds": after.cv_fold_scores,
            # Hold-out numbers are unchanged; show for reference
            "after_train_acc": after.train_accuracy,
            "after_test_acc": after.test_accuracy,
        })

    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print("  BEFORE vs AFTER — Hold-out split  vs  5-Fold Cross-Validation")
    print("=" * 90)
    header = (
        f"{'Node':>10}  "
        f"{'Formula':40}  "
        f"{'BEFORE Train':>12}  "
        f"{'BEFORE Test':>11}  "
        f"{'AFTER CV Mean':>13}  "
        f"{'AFTER CV Std':>12}  "
        f"{'AFTER Test':>10}"
    )
    print(header)
    print("-" * 90)
    for _, row in df.iterrows():
        fold_scores = row["after_cv_folds"]
        fold_str = "[" + " ".join(f"{s:.0%}" for s in fold_scores) + "]"
        print(
            f"{row['node_id']:>10}  "
            f"{row['formula']:40}  "
            f"{row['before_train_acc']:>12.1%}  "
            f"{row['before_test_acc']:>11.1%}  "
            f"{row['after_cv_mean']:>13.1%}  "
            f"{row['after_cv_std']:>12.1%}  "
            f"{row['after_test_acc']:>10.1%}"
        )
        print(f"{'':>68}  Fold scores: {fold_str}")
    print("-" * 90)

    # Summary delta
    delta = df["after_cv_mean"] - df["before_test_acc"]
    print(f"\nMean shift  (CV mean - hold-out test):  {delta.mean():+.1%}")
    print(f"  Positive = CV gives higher accuracy estimate")
    print(f"  Negative = hold-out was optimistic, CV is more conservative\n")


def main():
    state_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    if state_path is None:
        # Auto-pick the most recent run
        runs = sorted(
            Path("search_runs").glob("*/search_state.json"), key=lambda p: p.parent.name
        )
        if not runs:
            print("No search_state.json found under search_runs/. Run the search first.")
            sys.exit(1)
        state_path = runs[-1]

    print(f"Using: {state_path}")
    df = load_dataset(DATA_PATH)
    tmp_dir = Path("search_runs/_cv_compare_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    result_df = run_comparison(state_path, df, tmp_dir)

    if result_df.empty:
        print("No valid nodes found for comparison.")
        sys.exit(1)

    print_table(result_df)

    csv_out = state_path.parent / "cv_comparison.csv"
    save_cols = [c for c in result_df.columns if c != "after_cv_folds"]
    result_df[save_cols].to_csv(csv_out, index=False)
    print(f"CSV saved to {csv_out}")


if __name__ == "__main__":
    main()
