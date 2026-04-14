"""Combine top-N discovered descriptors as features in a Random Forest.

This script loads the best descriptor functions from one or more
search_state.json (MCTS) or pysr_results.json (PySR) files, computes
each descriptor's values on the full dataset, then trains a Random
Forest using those values as features.

This often pushes accuracy past 90% when each individual descriptor
stalls around 85-87% — the complementary information from multiple
descriptors is what the classifier needs.

Usage:
    uv run python run_ensemble.py
    uv run python run_ensemble.py --top_k 8 --n_estimators 500
    uv run python run_ensemble.py --runs search_runs/20260402_181719 search_runs/pysr_20260402_181331
"""

import argparse
import json
import logging
import signal
import textwrap
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from evaluator import load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger(__name__)

TIMEOUT_SECONDS = 10


class EvalTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise EvalTimeout("Function timed out")


# ---------------------------------------------------------------------------
# Load descriptor code strings from run directories
# ---------------------------------------------------------------------------

def _load_mcts_nodes(state_path: Path, top_k: int) -> list[dict]:
    """Return top_k nodes from a search_state.json by accuracy."""
    with state_path.open() as f:
        state = json.load(f)
    nodes = list(state["nodes"].values())
    nodes.sort(key=lambda n: n["accuracy"], reverse=True)
    results = []
    for n in nodes[:top_k]:
        if n.get("code", "").strip():
            results.append({
                "source": f"mcts:{state_path.parent.name}",
                "id": n["id"][:8],
                "accuracy": n["accuracy"],
                "code": n["code"],
                "formula": n.get("formula", ""),
            })
    return results


def _load_pysr_nodes(results_path: Path, top_k: int) -> list[dict]:
    """Return top_k formulas from a pysr_results.json by cv_accuracy."""
    with results_path.open() as f:
        data = json.load(f)
    formulas = data.get("top_formulas", [])
    results = []
    for r in formulas[:top_k]:
        if r.get("descriptor_code", "").strip():
            results.append({
                "source": f"pysr:{results_path.parent.name}",
                "id": f"pysr_{r['pysr_index']:03d}",
                "accuracy": r["cv_accuracy"],
                "code": r["descriptor_code"],
                "formula": r.get("formula_sympy", ""),
            })
    return results


def _discover_runs(search_dir: Path) -> tuple[list[Path], list[Path]]:
    """Auto-discover all MCTS and PySR run files under search_dir."""
    mcts_files = sorted(search_dir.glob("*/search_state.json"))
    pysr_files = sorted(search_dir.glob("pysr_*/pysr_results.json"))
    return mcts_files, pysr_files


# ---------------------------------------------------------------------------
# Execute descriptor code on dataset
# ---------------------------------------------------------------------------

def _exec_descriptor(code: str, df: pd.DataFrame) -> np.ndarray | None:
    """Run descriptor function on df; return None if it errors."""
    namespace = {"np": np, "math": __import__("math")}
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)
    try:
        exec(code, namespace)  # noqa: S102
        fn = namespace["descriptor"]
        values = []
        for _, row in df.iterrows():
            v = fn(
                rA=row["rA"], rB=row["rB"], rX=row["rX"],
                nA=row["nA"], nB=row["nB"], nX=row["nX"],
            )
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                return None
            values.append(float(v))
        return np.array(values)
    except Exception as e:
        log.debug("Descriptor failed: %s", e)
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ---------------------------------------------------------------------------
# Evaluate a feature matrix with Random Forest (5-fold CV)
# ---------------------------------------------------------------------------

def _rf_cv(X: np.ndarray, y: np.ndarray, n_estimators: int, cv_folds: int) -> tuple[float, float, list[float]]:
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring="accuracy")
    return float(scores.mean()), float(scores.std()), scores.tolist()


# ---------------------------------------------------------------------------
# Feature importance (train on full data)
# ---------------------------------------------------------------------------

def _feature_importance(X: np.ndarray, y: np.ndarray, names: list[str], n_estimators: int) -> list[tuple[str, float]]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    clf.fit(X_scaled, y)
    pairs = sorted(zip(names, clf.feature_importances_), key=lambda x: x[1], reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Random Forest ensemble of top descriptors")
    parser.add_argument("--runs", nargs="*", help="Specific run directories to load (default: auto-discover all)")
    parser.add_argument("--top_k", type=int, default=10, help="Top K descriptors to load per run (default: 10)")
    parser.add_argument("--n_estimators", type=int, default=300, help="RF trees (default: 300)")
    parser.add_argument("--cv_folds", type=int, default=5, help="CV folds (default: 5)")
    parser.add_argument("--data_path", default="perovskite-stability/TableS1.csv")
    parser.add_argument("--max_features", type=int, default=20,
                        help="Max feature columns to use (best by individual accuracy, default: 20)")
    args = parser.parse_args()

    df = load_dataset(args.data_path)
    y = df["exp_label"].values
    log.info("Loaded %d compounds", len(df))

    search_dir = Path("search_runs")

    # --- Collect candidate descriptor entries ---
    candidates: list[dict] = []

    if args.runs:
        run_dirs = [Path(r) for r in args.runs]
    else:
        mcts_files, pysr_files = _discover_runs(search_dir)
        run_dirs = list({p.parent for p in mcts_files + pysr_files})
        log.info("Auto-discovered %d run directories", len(run_dirs))

    for run_dir in run_dirs:
        state_file = run_dir / "search_state.json"
        pysr_file = run_dir / "pysr_results.json"
        if state_file.exists():
            nodes = _load_mcts_nodes(state_file, args.top_k)
            candidates.extend(nodes)
            log.info("  MCTS %s: loaded %d nodes", run_dir.name, len(nodes))
        if pysr_file.exists():
            nodes = _load_pysr_nodes(pysr_file, args.top_k)
            candidates.extend(nodes)
            log.info("  PySR %s: loaded %d formulas", run_dir.name, len(nodes))

    if not candidates:
        log.error("No candidates found. Check that search_runs/ contains valid run directories.")
        return

    log.info("Total candidates before deduplication: %d", len(candidates))

    # --- Compute feature values ---
    features: list[np.ndarray] = []
    feature_names: list[str] = []
    feature_accuracies: list[float] = []
    kept_candidates: list[dict] = []

    for c in candidates:
        vals = _exec_descriptor(c["code"], df)
        if vals is None:
            log.warning("  Skipping %s (execution error)", c["id"])
            continue
        if not np.isfinite(vals).all():
            log.warning("  Skipping %s (non-finite values)", c["id"])
            continue
        # Deduplicate by correlation (skip if >0.99 correlated with existing feature)
        if features:
            X_so_far = np.column_stack(features)
            corrs = np.abs(np.corrcoef(vals, X_so_far.T)[0, 1:])
            if corrs.max() > 0.99:
                log.info("  Skipping %s (duplicate of existing feature, corr=%.3f)", c["id"], corrs.max())
                continue
        features.append(vals)
        feature_names.append(f"{c['id']} ({c['source']}) acc={c['accuracy']:.3f}")
        feature_accuracies.append(c["accuracy"])
        kept_candidates.append(c)
        log.info("  Added feature: %s  individual_acc=%.3f", c["id"], c["accuracy"])

    if not features:
        log.error("No valid features found.")
        return

    # Sort by individual accuracy, keep top max_features
    order = np.argsort(feature_accuracies)[::-1][:args.max_features]
    features = [features[i] for i in order]
    feature_names = [feature_names[i] for i in order]
    kept_candidates = [kept_candidates[i] for i in order]

    log.info("Using %d features for ensemble", len(features))

    X_all = np.column_stack(features)

    # --- Full ensemble ---
    print("\n" + "=" * 70)
    print(f"Random Forest ensemble — {len(features)} features, {args.n_estimators} trees, {args.cv_folds}-fold CV")
    print("=" * 70)

    mean, std, fold_scores = _rf_cv(X_all, y, args.n_estimators, args.cv_folds)
    fold_str = "  [" + ", ".join(f"{s:.1%}" for s in fold_scores) + "]"
    print(f"\nFull ensemble ({len(features)} features):")
    print(f"  CV accuracy : {mean:.1%} ± {std:.1%}")
    print(f"  Fold scores : {fold_str}")

    # --- Feature importance ---
    print("\nFeature importances (full ensemble, trained on all data):")
    importances = _feature_importance(X_all, y, feature_names, args.n_estimators)
    for name, imp in importances:
        print(f"  {imp:.3f}  {name}")

    # --- Greedy forward selection: find best subset ---
    print("\n" + "-" * 70)
    print("Greedy forward feature selection:")
    best_subset_acc = 0.0
    best_subset: list[int] = []
    selected: list[int] = []
    remaining = list(range(len(features)))

    while remaining:
        best_acc_this_round = 0.0
        best_idx = None
        for i in remaining:
            trial = selected + [i]
            X_trial = np.column_stack([features[j] for j in trial])
            acc, _, _ = _rf_cv(X_trial, y, args.n_estimators, args.cv_folds)
            if acc > best_acc_this_round:
                best_acc_this_round = acc
                best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)
        log.info("  Step %d: added feature %d → CV=%.3f", len(selected), best_idx, best_acc_this_round)
        if best_acc_this_round > best_subset_acc:
            best_subset_acc = best_acc_this_round
            best_subset = selected[:]
        # Early stop if adding more features doesn't help
        if len(selected) >= 3 and best_acc_this_round < best_subset_acc - 0.005:
            log.info("  Stopping early (accuracy not improving)")
            break
        if len(selected) >= 8:
            break

    print(f"\nBest subset: {len(best_subset)} features, CV accuracy = {best_subset_acc:.1%}")
    X_best = np.column_stack([features[i] for i in best_subset])
    best_mean, best_std, best_folds = _rf_cv(X_best, y, args.n_estimators, args.cv_folds)
    fold_str = "  [" + ", ".join(f"{s:.1%}" for s in best_folds) + "]"
    print(f"  CV accuracy : {best_mean:.1%} ± {best_std:.1%}")
    print(f"  Fold scores : {fold_str}")
    print("\nSelected features:")
    for rank, i in enumerate(best_subset, 1):
        c = kept_candidates[i]
        print(f"  {rank}. [{c['id']}] (individual acc={c['accuracy']:.1%}) {c['source']}")
        formula_preview = c.get('formula', '')[:80]
        if formula_preview:
            print(f"     Formula: {formula_preview}")

    # --- Save results ---
    out_dir = search_dir / "ensemble_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ensemble_results.json"

    result = {
        "full_ensemble": {
            "n_features": len(features),
            "cv_accuracy": mean,
            "cv_std": std,
            "fold_scores": fold_scores,
        },
        "best_subset": {
            "n_features": len(best_subset),
            "cv_accuracy": best_mean,
            "cv_std": best_std,
            "fold_scores": best_folds,
            "features": [
                {
                    "rank": i + 1,
                    "id": kept_candidates[idx]["id"],
                    "source": kept_candidates[idx]["source"],
                    "individual_accuracy": kept_candidates[idx]["accuracy"],
                    "formula": kept_candidates[idx].get("formula", ""),
                    "code": kept_candidates[idx]["code"],
                }
                for i, idx in enumerate(best_subset)
            ],
        },
    }
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
