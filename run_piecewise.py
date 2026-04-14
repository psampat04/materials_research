"""Interpretable piecewise classifier for perovskite stability.

Uses physically motivated, constant-free features combined with a shallow
Decision Tree to produce explicit if/else rules — no magic numbers baked
into the features themselves.

Features (all derivable from first principles):
  t       Goldschmidt tolerance factor  (rA + rX) / (√2 · (rB + rX))
  τ       Bartel tau factor             rX/rB - nA·(nA - (rA/rB)/ln(rA/rB))
  μ       Octahedral factor             rB / rX
  χ       A/B charge ratio              nA / nB
  σ       A-to-anion size ratio         rA / rX
  δ       Charge-weighted size          (nA·rA + nB·rB) / (rA + rB)
  ν       Anion coordination tendency   nA·nB / nX²  (charge balance)
  ρ       B-site size relative to A     rB / rA

The Decision Tree learns piecewise thresholds on these 8 features.
Accuracy is estimated with 5×3 nested cross-validation (outer loop tests
the model, inner loop selects tree depth) to avoid selection bias.

A held-out test accuracy is also computed using the pre-assigned is_train
column in the dataset (same split used by LLMSR / DSR / FEX).

Usage:
    uv run python run_piecewise.py
    uv run python run_piecewise.py --max_depth 5 --outer_folds 5
"""

import argparse
import json
import logging
import math
import textwrap
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text

from evaluator import load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physically motivated, constant-free features
# ---------------------------------------------------------------------------

FEATURE_DEFS = {
    "t":   "Goldschmidt tolerance  (rA+rX) / (√2·(rB+rX))",
    "tau": "Bartel tau             rX/rB - nA·(nA - (rA/rB)/ln(rA/rB))",
    "mu":  "Octahedral factor      rB / rX",
    "chi": "A/B charge ratio       nA / nB",
    "sig": "A-to-anion size ratio  rA / rX",
    "dlt": "Charge-weighted size   (nA·rA + nB·rB) / (rA + rB)",
    "nu":  "Charge balance         nA·nB / nX²",
    "rho": "B/A size ratio         rB / rA",
}


def compute_features(df: pd.DataFrame) -> np.ndarray:
    rA = df["rA"].values
    rB = df["rB"].values
    rX = df["rX"].values
    nA = df["nA"].values.astype(float)
    nB = df["nB"].values.astype(float)
    nX = df["nX"].values.astype(float)

    # Goldschmidt tolerance factor
    t = (rA + rX) / (math.sqrt(2) * (rB + rX))

    # Bartel tau (clip log arg for numerical safety; rA > rB guaranteed)
    log_ratio = np.log(np.clip(rA / rB, 1e-9, None))
    tau = rX / rB - nA * (nA - (rA / rB) / log_ratio)

    # Octahedral factor
    mu = rB / rX

    # A/B charge ratio
    chi = nA / nB

    # A-to-anion size ratio
    sig = rA / rX

    # Charge-weighted mean radius
    dlt = (nA * rA + nB * rB) / (rA + rB)

    # Charge balance proxy
    nu = nA * nB / (nX ** 2)

    # B-to-A size ratio
    rho = rB / rA

    return np.column_stack([t, tau, mu, chi, sig, dlt, nu, rho])


FEATURE_NAMES = list(FEATURE_DEFS.keys())


# ---------------------------------------------------------------------------
# Export decision tree as human-readable if/else rules
# ---------------------------------------------------------------------------

def tree_to_rules(clf: DecisionTreeClassifier, feature_names: list[str]) -> str:
    return export_text(clf, feature_names=feature_names, decimals=4)


def tree_to_python(clf: DecisionTreeClassifier, feature_names: list[str]) -> str:
    """Generate a standalone Python function from a fitted decision tree."""
    tree = clf.tree_
    lines = [
        "def predict_perovskite(rA, rB, rX, nA, nB, nX):",
        '    """Return 1 (perovskite) or -1 (non-perovskite)."""',
        "    import math, numpy as np",
        "    log_ratio = math.log(max(rA / rB, 1e-9))",
        "    t   = (rA + rX) / (math.sqrt(2) * (rB + rX))",
        "    tau = rX/rB - nA * (nA - (rA/rB) / log_ratio)",
        "    mu  = rB / rX",
        "    chi = nA / nB",
        "    sig = rA / rX",
        "    dlt = (nA * rA + nB * rB) / (rA + rB)",
        "    nu  = nA * nB / (nX ** 2)",
        "    rho = rB / rA",
        f"    X   = [{', '.join(feature_names)}]",
        "",
    ]

    def recurse(node: int, depth: int) -> list[str]:
        indent = "    " + "    " * depth
        if tree.feature[node] == -2:
            # leaf
            counts = tree.value[node][0]
            pred = clf.classes_[np.argmax(counts)]
            total = int(counts.sum())
            perov = int(counts[list(clf.classes_).index(1)])
            return [f"{indent}return {int(pred)}  # {perov}/{total} train samples"]
        feat = feature_names[tree.feature[node]]
        thresh = tree.threshold[node]
        idx = feature_names.index(feat)
        result = [
            f"{indent}if X[{idx}] <= {thresh:.4f}:  # {feat} <= {thresh:.4f}",
        ]
        result += recurse(tree.children_left[node], depth + 1)
        result += [f"{indent}else:  # {feat} > {thresh:.4f}"]
        result += recurse(tree.children_right[node], depth + 1)
        return result

    lines += recurse(0, 0)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="perovskite-stability/TableS1.csv")
    parser.add_argument("--max_depth", type=int, default=5,
                        help="Max tree depth to search (default: 5)")
    parser.add_argument("--outer_folds", type=int, default=5)
    parser.add_argument("--inner_folds", type=int, default=3)
    args = parser.parse_args()

    df = load_dataset(args.data_path)
    y = df["exp_label"].values
    log.info("Loaded %d compounds", len(df))

    X = compute_features(df)
    log.info("Computed %d features per compound", X.shape[1])

    # ------------------------------------------------------------------
    # Feature overview
    # ------------------------------------------------------------------
    print("\nPhysical features (no fitted constants):")
    for i, (name, desc) in enumerate(FEATURE_DEFS.items()):
        vals = X[:, i]
        print(f"  {name:4s}  {desc}  [range {vals.min():.3f}–{vals.max():.3f}]")

    # ------------------------------------------------------------------
    # Nested cross-validation (outer: accuracy estimate, inner: depth selection)
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"Nested CV  (outer={args.outer_folds}-fold, inner={args.inner_folds}-fold)")
    print("=" * 65)

    outer_cv = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=0)

    param_grid = {"max_depth": list(range(2, args.max_depth + 1))}
    base_clf = DecisionTreeClassifier(class_weight="balanced", random_state=42)
    grid_clf = GridSearchCV(base_clf, param_grid, cv=inner_cv, scoring="accuracy", refit=True)

    outer_scores = cross_val_score(grid_clf, X, y, cv=outer_cv, scoring="accuracy")

    fold_str = "  [" + ", ".join(f"{s:.1%}" for s in outer_scores) + "]"
    print(f"Nested CV accuracy: {outer_scores.mean():.1%} ± {outer_scores.std():.1%}")
    print(f"Fold scores       : {fold_str}")

    # ------------------------------------------------------------------
    # Refit on full data to get final tree + rules
    # ------------------------------------------------------------------
    grid_clf.fit(X, y)
    best_depth = grid_clf.best_params_["max_depth"]
    final_clf = grid_clf.best_estimator_
    train_acc = (final_clf.predict(X) == y).mean()

    print(f"\nBest tree depth (inner CV): {best_depth}")
    print(f"Training accuracy (full data): {train_acc:.1%}")

    # ------------------------------------------------------------------
    # Print the tree rules
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Decision tree rules (piecewise classifier):")
    print("=" * 65)
    print(tree_to_rules(final_clf, FEATURE_NAMES))

    # ------------------------------------------------------------------
    # Print feature importances
    # ------------------------------------------------------------------
    print("Feature importances:")
    importances = sorted(
        zip(FEATURE_NAMES, final_clf.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    for name, imp in importances:
        bar = "█" * int(imp * 40)
        print(f"  {name:4s}  {imp:.3f}  {bar}  {FEATURE_DEFS[name]}")

    # ------------------------------------------------------------------
    # Print the standalone Python function
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Standalone Python predict function:")
    print("=" * 65)
    print(tree_to_python(final_clf, FEATURE_NAMES))

    # ------------------------------------------------------------------
    # Per-anion breakdown on full data
    # ------------------------------------------------------------------
    preds = final_clf.predict(X)
    print("\nPer-anion accuracy (trained on full data):")
    for anion in ["O", "F", "Cl", "Br", "I"]:
        mask = df["X"].values == anion
        if mask.any():
            acc = (preds[mask] == y[mask]).mean()
            print(f"  {anion:2s}: {acc:.1%}  ({mask.sum()} compounds)")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_dir = Path("search_runs/piecewise_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    rules_path = out_dir / "tree_rules.txt"
    rules_path.write_text(tree_to_rules(final_clf, FEATURE_NAMES))

    code_path = out_dir / "predict_function.py"
    code_path.write_text(
        "# Auto-generated piecewise classifier — no fitted constants in features\n"
        "# Features are physically motivated (Goldschmidt t, Bartel tau, etc.)\n"
        "# Tree thresholds were learned from 576 experimentally characterized ABX3 compounds\n\n"
        + tree_to_python(final_clf, FEATURE_NAMES)
        + "\n"
    )

    print(f"\nRules saved to  : {rules_path}")
    print(f"Function saved  : {code_path}")
    print(f"\nNested CV accuracy (unbiased estimate): {outer_scores.mean():.1%} ± {outer_scores.std():.1%}")

    # ------------------------------------------------------------------
    # Held-out test accuracy using the pre-assigned is_train split
    # (same split used by LLMSR, DSR, and FEX for fair comparison)
    # ------------------------------------------------------------------
    if "is_train" in df.columns:
        train_mask = df["is_train"].values == 1
        test_mask  = ~train_mask

        X_train_ho = X[train_mask]
        y_train_ho = y[train_mask]
        X_test_ho  = X[test_mask]
        y_test_ho  = y[test_mask]

        # Use same best depth from inner CV
        ho_clf = DecisionTreeClassifier(
            max_depth=best_depth, class_weight="balanced", random_state=42
        )
        ho_clf.fit(X_train_ho, y_train_ho)

        ho_train_acc = float((ho_clf.predict(X_train_ho) == y_train_ho).mean())
        ho_test_acc  = float((ho_clf.predict(X_test_ho)  == y_test_ho).mean())

        print("\n" + "=" * 65)
        print("Held-out test accuracy  (is_train split — same as LLMSR/DSR/FEX):")
        print("=" * 65)
        print(f"  Train accuracy : {ho_train_acc:.1%}  ({train_mask.sum()} compounds)")
        print(f"  Test  accuracy : {ho_test_acc:.1%}  ({test_mask.sum()} compounds)")

        # Per-anion breakdown on test set
        print("\nPer-anion test accuracy:")
        for anion in ["O", "F", "Cl", "Br", "I"]:
            mask = (df["X"].values == anion) & test_mask
            if mask.any():
                acc = (ho_clf.predict(X[mask]) == y[mask]).mean()
                print(f"  {anion:2s}: {acc:.1%}  ({mask.sum()} test compounds)")

        # Save JSON result for benchmark_comparison.py
        fem_result = {
            "method":          "FEM",
            "cv_accuracy":     float(outer_scores.mean()),
            "cv_std":          float(outer_scores.std()),
            "cv_fold_scores":  outer_scores.tolist(),
            "train_accuracy":  ho_train_acc,
            "test_accuracy":   ho_test_acc,
            "best_tree_depth": best_depth,
            "n_features":      len(FEATURE_NAMES),
            "feature_names":   FEATURE_NAMES,
        }
        fem_path = out_dir / "fem_results.json"
        with open(fem_path, "w") as fh:
            json.dump(fem_result, fh, indent=2)
        print(f"\nFEM results saved → {fem_path}")


if __name__ == "__main__":
    main()
