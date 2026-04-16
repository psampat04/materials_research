"""Parameter optimization for LLM-SR descriptor skeletons."""

import re

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

MAX_PARAMS = 2


def count_params(code: str) -> int:
    """Count how many params[] slots the skeleton uses, capped at MAX_PARAMS."""
    indices = re.findall(r"params\[(\d+)\]", code)
    if not indices:
        return 0
    return min(max(int(i) for i in indices) + 1, MAX_PARAMS)


def bake_params(skeleton_code: str, params: np.ndarray) -> str:
    """Inject fitted params into the skeleton, returning a standard 6-arg descriptor."""
    params_list = [float(p) for p in params]
    baked = re.sub(
        r"def descriptor\(rA,\s*rB,\s*rX,\s*nA,\s*nB,\s*nX,\s*params\s*\):",
        f"def descriptor(rA, rB, rX, nA, nB, nX):\n    params = {params_list}",
        skeleton_code,
    )
    return baked


def fit_params(skeleton_code: str, df: pd.DataFrame, n_params: int) -> tuple[np.ndarray, float]:
    """Fit params[] using Nelder-Mead to maximise logistic-regression accuracy.

    Returns (best_params, best_accuracy).
    """
    namespace = {"np": np, "math": __import__("math")}
    try:
        exec(skeleton_code, namespace)  # noqa: S102
    except SyntaxError as e:
        raise ValueError(f"Skeleton has syntax error: {e}") from e
    fn = namespace["descriptor"]
    labels = df["exp_label"].values

    def objective(params: np.ndarray) -> float:
        try:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                values = np.array([
                    fn(
                        rA=row["rA"], rB=row["rB"], rX=row["rX"],
                        nA=row["nA"], nB=row["nB"], nX=row["nX"],
                        params=params,
                    )
                    for _, row in df.iterrows()
                ])
            finite_mask = np.isfinite(values)
            if finite_mask.sum() < len(values) * 0.8:
                return 1.0
            # Replace remaining non-finite with median of finite values
            values = np.where(finite_mask, values, np.median(values[finite_mask]))
            lr = LogisticRegression(max_iter=200, solver="lbfgs")
            lr.fit(values.reshape(-1, 1), labels)
            return -lr.score(values.reshape(-1, 1), labels)
        except Exception:
            return 1.0

    rng = np.random.default_rng(42)
    starting_points = [
        np.ones(n_params),
        np.full(n_params, 0.5),
        np.full(n_params, 2.0),
        np.full(n_params, -1.0),
        rng.uniform(0.1, 3.0, n_params),
        rng.uniform(-2.0, 2.0, n_params),
        rng.standard_normal(n_params) * 0.5,
        rng.uniform(0.01, 0.5, n_params),
    ]

    best_result = None
    for x0 in starting_points:
        result = minimize(
            objective,
            x0=x0,
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-5},
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    return best_result.x, -best_result.fun
