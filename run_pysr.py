"""Run PySR symbolic regression to find perovskite stability descriptors.

This is a standalone alternative to MCTS that uses evolutionary symbolic
regression (no LLM). It searches for 1-D descriptor expressions that
maximise 5-fold cross-validated accuracy when classified by a decision tree.

Usage:
    uv run python run_pysr.py
    uv run python run_pysr.py pysr.niterations=100 pysr.populations=30
"""

import json
import logging
import textwrap
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import sympy
from omegaconf import DictConfig
from pysr import PySRRegressor

from evaluator import evaluate_candidate, load_dataset

log = logging.getLogger(__name__)

FEATURE_COLS = ["rA", "rB", "rX", "nA", "nB", "nX"]


# ---------------------------------------------------------------------------
# Expression → descriptor code string
# ---------------------------------------------------------------------------

def _expr_to_code(expr: sympy.Expr) -> str:
    """Convert a sympy expression to a Python descriptor function string."""
    py_expr = sympy.pycode(expr)
    return textwrap.dedent(f"""\
        import math
        import numpy as np

        def descriptor(rA, rB, rX, nA, nB, nX):
            return float({py_expr})
    """)


# ---------------------------------------------------------------------------
# Goldschmidt / Bartel-tau checks (mirrors the MCTS prompt constraints)
# ---------------------------------------------------------------------------

_GOLDSCHMIDT_VARS = {"rA", "rB", "rX"}
_TAU_PATTERN_NODES = {"log", "Mul", "Add"}  # rough structural check


def _is_forbidden(expr: sympy.Expr) -> bool:
    """Return True if the expression is a trivial rescaling of t or tau."""
    free = {str(s) for s in expr.free_symbols}
    if not free.issubset(set(FEATURE_COLS)):
        return True  # uses unknown variables

    # Build goldschmidt t and tau as sympy expressions for comparison
    rA, rB, rX, nA, nB, nX = sympy.symbols("rA rB rX nA nB nX")
    t = (rA + rX) / (sympy.sqrt(2) * (rB + rX))
    tau = rX / rB - nA * (nA - (rA / rB) / sympy.log(rA / rB))

    # Check structural similarity by ratio simplification
    for ref in (t, tau):
        ratio = sympy.simplify(expr / ref)
        if ratio.free_symbols == set():  # ratio is a plain number → rescaling
            return True
    return False


# ---------------------------------------------------------------------------
# Evaluate all equations returned by PySR
# ---------------------------------------------------------------------------

def _evaluate_equations(
    equations: pd.DataFrame,
    df: pd.DataFrame,
    plot_dir: Path,
    cfg: DictConfig,
) -> list[dict]:
    results = []
    dt_depth = cfg.eval.decision_tree_max_depth
    train_label = cfg.eval.train_split_label
    cv_folds = cfg.eval.cv_folds

    for idx, row in equations.iterrows():
        expr = row.get("sympy_format")
        if expr is None or not isinstance(expr, sympy.Basic):
            continue

        node_id = f"pysr_{idx:03d}"

        if _is_forbidden(expr):
            log.info("Skipping forbidden expression %s: %s", node_id, expr)
            continue

        code = _expr_to_code(expr)
        log.info("Evaluating %s: %s", node_id, expr)

        result = evaluate_candidate(
            func_code=code,
            df=df,
            plot_dir=plot_dir,
            node_id=node_id,
            decision_tree_max_depth=dt_depth,
            train_split_label=train_label,
            cv_folds=cv_folds,
        )

        if result.error:
            log.warning("  Error: %s", result.error)
            continue

        results.append({
            "pysr_index": int(idx),
            "formula_sympy": str(expr),
            "formula_latex": sympy.latex(expr),
            "complexity": int(row.get("complexity", -1)),
            "pysr_loss": float(row.get("loss", float("nan"))),
            "cv_accuracy": result.cv_accuracy,
            "cv_std": result.cv_std,
            "cv_fold_scores": result.cv_fold_scores,
            "train_accuracy": result.train_accuracy,
            "test_accuracy": result.test_accuracy,
            "false_positive_rate": result.false_positive_rate,
            "per_anion_accuracy": result.per_anion_accuracy,
            "plot_path": result.plot_path,
            "descriptor_code": code,
            "metrics_summary": result.metrics_summary,
        })
        log.info(
            "  CV: %.1f%% ± %.1f%% | train: %.1f%% | test: %.1f%%",
            result.cv_accuracy * 100,
            result.cv_std * 100,
            result.train_accuracy * 100,
            result.test_accuracy * 100,
        )

    results.sort(key=lambda r: r["cv_accuracy"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------

def _print_results(results: list[dict], k: int = 10) -> None:
    print("\n" + "=" * 70)
    print(f"Top {min(k, len(results))} PySR formulas by 5-fold CV accuracy")
    print("=" * 70)
    for i, r in enumerate(results[:k], 1):
        print(f"\n--- #{i} (PySR index {r['pysr_index']}, complexity={r['complexity']}) ---")
        print(f"  CV accuracy : {r['cv_accuracy']:.1%} ± {r['cv_std']:.1%}")
        folds = "  [" + ", ".join(f"{s:.1%}" for s in r["cv_fold_scores"]) + "]"
        print(f"  Fold scores : {folds}")
        print(f"  Train / Test: {r['train_accuracy']:.1%} / {r['test_accuracy']:.1%}")
        pa = r["per_anion_accuracy"]
        if pa:
            anion_str = ", ".join(f"{a}={v:.0%}" for a, v in pa.items())
            print(f"  Per-anion   : {anion_str}")
        print(f"  Formula     : {r['formula_sympy']}")
        print(f"  LaTeX       : {r['formula_latex']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="conf", config_name="pysr_config", version_base=None)
def main(cfg: DictConfig) -> None:
    orig_cwd = hydra.utils.get_original_cwd()
    data_path = Path(orig_cwd) / cfg.eval.data_path

    run_dir = (
        Path(orig_cwd)
        / cfg.search.output_dir
        / f"pysr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    log.info("Run directory: %s", run_dir)

    df = load_dataset(str(data_path))
    log.info("Loaded %d compounds from %s", len(df), data_path)

    X = df[FEATURE_COLS].values
    y = df["exp_label"].values.astype(float)

    # ------------------------------------------------------------------
    # Configure PySR
    # ------------------------------------------------------------------
    pysr_kwargs = dict(
        niterations=cfg.pysr.niterations,
        populations=cfg.pysr.populations,
        population_size=cfg.pysr.population_size,
        ncycles_per_iteration=cfg.pysr.ncycles_per_iteration,
        maxsize=cfg.pysr.maxsize,
        binary_operators=list(cfg.pysr.binary_operators),
        unary_operators=list(cfg.pysr.unary_operators),
        parsimony=cfg.pysr.parsimony,
        random_state=cfg.pysr.random_state,
        # Suppress verbose Julia output — still logs via Python logger
        verbosity=0,
        # Temp dir for Julia artifacts (avoids cluttering workspace)
        tempdir=str(run_dir / "_julia_tmp"),
    )

    log.info("Starting PySR search (niterations=%d, populations=%d)", cfg.pysr.niterations, cfg.pysr.populations)
    model = PySRRegressor(**pysr_kwargs)
    model.fit(X, y, variable_names=FEATURE_COLS)

    equations: pd.DataFrame = model.equations_
    log.info("PySR found %d candidate equations", len(equations))

    # ------------------------------------------------------------------
    # Evaluate each expression via the same pipeline as MCTS
    # ------------------------------------------------------------------
    results = _evaluate_equations(equations, df, plot_dir, cfg)

    if not results:
        log.warning("No valid equations passed evaluation — try relaxing constraints.")
        return

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_path = run_dir / "pysr_results.json"
    with results_path.open("w") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "n_equations_found": len(equations),
                "n_evaluated": len(results),
                "top_formulas": results,
            },
            f,
            indent=2,
        )
    log.info("Results saved to %s", results_path)

    _print_results(results)

    print(f"\nSearch complete.")
    print(f"  Equations found by PySR : {len(equations)}")
    print(f"  Successfully evaluated  : {len(results)}")
    print(f"  Best CV accuracy        : {results[0]['cv_accuracy']:.1%} ± {results[0]['cv_std']:.1%}")
    print(f"  Best formula            : {results[0]['formula_sympy']}")
    print(f"  Results saved to        : {results_path}")


if __name__ == "__main__":
    main()
