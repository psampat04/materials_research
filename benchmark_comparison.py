"""Benchmark comparison: FEX vs LLMSR vs DSR vs FEM on perovskite stability.

Reads held-out test accuracies from each method's saved results and prints
a side-by-side comparison table.  Only test accuracy (never CV or train) is
used for ranking.

Methods
-------
FEX   (run_fex.py)
    RL over binary expression trees (adapted from FEX-PIDEs-PDEs).
    Result file: search_runs/fex_results/fex_results.json

LLMSR (run_search.py)
    LLM-guided MCTS symbolic-descriptor search.
    Result files: search_runs/<timestamp>/search_state.json  (all runs)

DSR   (run_pysr.py)
    Direct/Deep Symbolic Regression via PySR.
    Result files: search_runs/pysr_<timestamp>/pysr_results.json  (all runs)

FEM   (run_piecewise.py)
    Feature Engineering Method — physics-motivated features + Decision Tree.
    Result file: search_runs/piecewise_results/fem_results.json

Baselines from Bartel et al. (2019) are included for reference:
    Goldschmidt t  ~74 %
    Bartel τ       ~92 %

Usage:
    uv run python benchmark_comparison.py
    uv run python benchmark_comparison.py --search_runs search_runs
"""

import argparse
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_fex(search_runs: Path) -> dict | None:
    path = search_runs / "fex_results" / "fex_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    return {
        "method":          "FEX",
        "description":     "RL over expression trees (FEX-PIDEs-PDEs adapted)",
        "test_accuracy":   d["test_accuracy"],
        "train_accuracy":  d["train_accuracy"],
        "best_formula":    d.get("best_expression", "—"),
        "source":          str(path),
    }


def load_llmsr(search_runs: Path) -> dict | None:
    best: dict | None = None
    for state_file in sorted(search_runs.glob("*/search_state.json")):
        try:
            with open(state_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for node in data.get("nodes", {}).values():
            ta = node.get("metrics", {}).get("test_accuracy", 0.0)
            if best is None or ta > best["test_accuracy"]:
                best = {
                    "method":         "LLMSR",
                    "description":    "LLM-guided MCTS symbolic regression",
                    "test_accuracy":  ta,
                    "train_accuracy": node["metrics"].get("train_accuracy", 0.0),
                    "best_formula":   node.get("formula", "—")[:80],
                    "source":         str(state_file),
                    "run":            state_file.parent.name,
                }
    return best


def load_dsr(search_runs: Path) -> dict | None:
    best: dict | None = None
    for result_file in sorted(search_runs.glob("pysr_*/pysr_results.json")):
        try:
            with open(result_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for formula in data.get("top_formulas", []):
            ta = formula.get("test_accuracy", 0.0)
            if best is None or ta > best["test_accuracy"]:
                best = {
                    "method":         "DSR",
                    "description":    "Direct Symbolic Regression (PySR)",
                    "test_accuracy":  ta,
                    "train_accuracy": formula.get("train_accuracy", 0.0),
                    "best_formula":   formula.get("formula_sympy", "—")[:80],
                    "source":         str(result_file),
                    "run":            result_file.parent.name,
                }
    return best


def load_fem(search_runs: Path) -> dict | None:
    path = search_runs / "piecewise_results" / "fem_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    return {
        "method":          "FEM",
        "description":     "Feature Engineering Method (physics features + Decision Tree)",
        "test_accuracy":   d["test_accuracy"],
        "train_accuracy":  d["train_accuracy"],
        "best_formula":    f"Decision Tree (depth={d.get('best_tree_depth','?')}) on {d.get('n_features','?')} physics features",
        "source":          str(path),
    }


# ---------------------------------------------------------------------------
# Bartel et al. (2019) baselines (published numbers)
# ---------------------------------------------------------------------------

LITERATURE_BASELINES = [
    {
        "method":         "Goldschmidt t (baseline)",
        "description":    "Goldschmidt tolerance factor  t = (rA+rX)/(√2·(rB+rX))",
        "test_accuracy":  0.7396,
        "train_accuracy": None,
        "best_formula":   "t = (rA + rX) / (√2 · (rB + rX))",
        "source":         "Bartel et al. (2019) Table S1 — published 74% accuracy",
    },
    {
        "method":         "Bartel τ (target)",
        "description":    "Bartel tau factor  τ = rX/rB - nA·(nA - (rA/rB)/ln(rA/rB))",
        "test_accuracy":  0.9201,
        "train_accuracy": None,
        "best_formula":   "τ = rX/rB − nA·(nA − (rA/rB)/ln(rA/rB))",
        "source":         "Bartel et al. (2019) — published 92% accuracy",
    },
]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _bar(acc: float, width: int = 30) -> str:
    filled = round(acc * width)
    return "█" * filled + "░" * (width - filled)


def _fmt_acc(acc: float | None) -> str:
    if acc is None:
        return "    —    "
    return f"  {acc:.1%}  "


def print_table(results: list[dict]) -> None:
    results_sorted = sorted(results, key=lambda r: r["test_accuracy"], reverse=True)

    col_method = 26
    col_acc    = 10
    col_bar    = 32

    sep = "─" * (col_method + col_acc + col_bar + 8)

    print()
    print("=" * len(sep))
    print("  Perovskite Stability Benchmark  —  Held-out TEST Accuracy")
    print("=" * len(sep))
    print(
        f"{'Method':<{col_method}}  {'Test Acc':>{col_acc}}  "
        f"{'Progress bar':<{col_bar}}"
    )
    print(sep)

    for r in results_sorted:
        ta   = r["test_accuracy"]
        bar  = _bar(ta, col_bar - 2)
        flag = "  ◄ best" if r == results_sorted[0] else ""
        print(
            f"  {r['method']:<{col_method - 2}}  {ta:>8.1%}  {bar}{flag}"
        )

    print(sep)
    print()

    # Detailed rows
    print("─" * 70)
    print("Detail")
    print("─" * 70)
    for r in results_sorted:
        ta_str  = f"{r['test_accuracy']:.4f} ({r['test_accuracy']:.1%})"
        tra_str = f"{r['train_accuracy']:.4f} ({r['train_accuracy']:.1%})" \
                  if r.get("train_accuracy") is not None else "—"
        print(f"\n  {'Method':<14}: {r['method']}")
        print(f"  {'Description':<14}: {r['description']}")
        print(f"  {'Test  accuracy':<14}: {ta_str}")
        print(f"  {'Train accuracy':<14}: {tra_str}")
        print(f"  {'Best formula':<14}: {r['best_formula'][:72]}")
        print(f"  {'Source':<14}: {r['source']}")

    print()
    print("─" * 70)
    print("Note: test accuracy uses the held-out split (is_train=0 in TableS1.csv).")
    print("      CV and training accuracies are excluded from the ranking.")
    print("─" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_runs", default="search_runs",
                        help="Path to the search_runs directory")
    parser.add_argument("--include_baselines", action="store_true", default=True,
                        help="Include Goldschmidt t and Bartel τ literature baselines")
    args = parser.parse_args()

    runs_dir = Path(args.search_runs)

    results: list[dict] = []
    missing: list[str] = []

    loaders = [
        ("FEX",   load_fex),
        ("LLMSR", load_llmsr),
        ("DSR",   load_dsr),
        ("FEM",   load_fem),
    ]

    for name, loader in loaders:
        r = loader(runs_dir)
        if r is not None:
            results.append(r)
            print(f"[✓] {name:6s} — test_accuracy = {r['test_accuracy']:.1%}")
        else:
            missing.append(name)
            print(f"[✗] {name:6s} — no results found (run the corresponding script first)")

    if args.include_baselines:
        results.extend(LITERATURE_BASELINES)

    if not results:
        print("\nNo results available yet. Run at least one method first.")
        return

    if missing:
        print(f"\n  Missing methods: {', '.join(missing)}")
        print("  The table below shows only methods with available results.\n")

    print_table(results)

    # Save combined JSON
    out_path = runs_dir / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "methods": results,
                "ranking": [
                    {"rank": i + 1, "method": r["method"], "test_accuracy": r["test_accuracy"]}
                    for i, r in enumerate(
                        sorted(results, key=lambda r: r["test_accuracy"], reverse=True)
                    )
                ],
            },
            f,
            indent=2,
        )
    print(f"Saved combined benchmark → {out_path}")


if __name__ == "__main__":
    main()
