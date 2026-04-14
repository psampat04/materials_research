"""FEX-style RL over expression trees for perovskite stability descriptor discovery.

Adapts the Function EXpression (FEX) approach from the FEX-PIDEs-PDEs codebase
to the perovskite stability classification problem.

FEX core idea (from fex/Poisson6.1.3a/controller_PIDESv2.py):
  - Maintain a fixed binary tree topology.
  - An RL controller (REINFORCE) selects an operator for every tree node.
  - Each candidate formula is evaluated; the reward drives policy updates.
  - Only the RL search (operator selection) is trained here — leaves are the
    raw atomic features (rA, rB, rX, nA, nB, nX) instead of neural networks.

Tree topology used (FEX "depth2_sub"):

    [root_unary]
         |
    [binary_op]
    /           \\
[left_unary]  [right_unary]
    |                |
 [feat_i]        [feat_j]

Resulting expression: root_unary( binary_op( left_unary(feat_i), right_unary(feat_j) ) )

Action sequence (in-order traversal, 6 positions):
  0: feat_left   — which input feature for the left leaf   (N_FEATURES choices)
  1: left_unary  — unary transform on the left leaf         (N_UNARY   choices)
  2: binary_op   — binary operator at the inner node        (N_BINARY  choices)
  3: feat_right  — which input feature for the right leaf   (N_FEATURES choices)
  4: right_unary — unary transform on the right leaf        (N_UNARY   choices)
  5: root_unary  — outer unary transform on the whole expr  (N_UNARY   choices)

The RL controller maintains one softmax logit vector per tree-node position
and is updated with REINFORCE (policy gradient), implemented in pure NumPy —
no additional dependencies beyond what the project already uses.

Usage:
    uv run python run_fex.py
    uv run python run_fex.py --epochs 1000 --batch_size 32
"""

import argparse
import json
import logging
import random
import warnings
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from evaluator import load_dataset

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Operator and feature sets  (mirrors function.py from FEX-PIDEs-PDEs)
# ---------------------------------------------------------------------------

FEATURES   = ["rA", "rB", "rX", "nA", "nB", "nX"]
N_FEATURES = len(FEATURES)

_UNARY_OPS: list[tuple[str, object]] = [
    ("identity", lambda x: x),
    ("abs",      lambda x: np.abs(x)),
    ("sqrt",     lambda x: np.sqrt(np.maximum(np.abs(x), 0.0))),
    ("log",      lambda x: np.log(np.maximum(np.abs(x), 1e-9))),
    ("square",   lambda x: x ** 2),
    ("neg",      lambda x: -x),
    ("inv",      lambda x: 1.0 / (np.abs(x) + 1e-9)),
    ("cube",     lambda x: x ** 3),
]
UNARY_NAMES = [n for n, _ in _UNARY_OPS]
UNARY_FNS   = [f for _, f in _UNARY_OPS]
N_UNARY     = len(_UNARY_OPS)

_BINARY_OPS: list[tuple[str, object]] = [
    ("add", lambda x, y: x + y),
    ("sub", lambda x, y: x - y),
    ("mul", lambda x, y: x * y),
    ("div", lambda x, y: x / (np.abs(y) + 1e-9)),
    ("max", lambda x, y: np.maximum(x, y)),
    ("min", lambda x, y: np.minimum(x, y)),
]
BINARY_NAMES = [n for n, _ in _BINARY_OPS]
BINARY_FNS   = [f for _, f in _BINARY_OPS]
N_BINARY     = len(_BINARY_OPS)

# Action slot cardinalities (indices 0-5 defined in module docstring)
ACTION_SIZES = [N_FEATURES, N_UNARY, N_BINARY, N_FEATURES, N_UNARY, N_UNARY]


# ---------------------------------------------------------------------------
# Expression tree: evaluate and stringify
# ---------------------------------------------------------------------------

def eval_expression(
    actions: list[int],
    feature_vals: dict[str, np.ndarray],
) -> np.ndarray:
    """Evaluate the expression tree for all compounds."""
    feat_l, u_l, bin_op, feat_r, u_r, u_root = actions
    left   = UNARY_FNS[u_l](feature_vals[FEATURES[feat_l]])
    right  = UNARY_FNS[u_r](feature_vals[FEATURES[feat_r]])
    inner  = BINARY_FNS[bin_op](left, right)
    result = UNARY_FNS[u_root](inner)
    if not np.isfinite(result).all():
        raise ValueError("non-finite descriptor values")
    return result


def _u_str(name: str, arg: str) -> str:
    return arg if name == "identity" else f"{name}({arg})"


def _b_str(name: str, l: str, r: str) -> str:
    sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
    if name in sym:
        return f"({l} {sym[name]} {r})"
    return f"{name}({l}, {r})"


def actions_to_expr(actions: list[int]) -> str:
    feat_l, u_l, bin_op, feat_r, u_r, u_root = actions
    l   = _u_str(UNARY_NAMES[u_l],   FEATURES[feat_l])
    r   = _u_str(UNARY_NAMES[u_r],   FEATURES[feat_r])
    mid = _b_str(BINARY_NAMES[bin_op], l, r)
    return _u_str(UNARY_NAMES[u_root], mid)


def actions_to_code(actions: list[int]) -> str:
    feat_l, u_l, bin_op, feat_r, u_r, u_root = actions

    _uc = {
        "identity": "{}",
        "abs":      "np.abs({})",
        "sqrt":     "np.sqrt(np.maximum(np.abs({}), 0.0))",
        "log":      "np.log(np.maximum(np.abs({}), 1e-9))",
        "square":   "({}) ** 2",
        "neg":      "-({})",
        "inv":      "1.0 / (np.abs({}) + 1e-9)",
        "cube":     "({}) ** 3",
    }
    _bc = {
        "add": "({} + {})",
        "sub": "({} - {})",
        "mul": "({} * {})",
        "div": "({} / (np.abs({}) + 1e-9))",
        "max": "np.maximum({}, {})",
        "min": "np.minimum({}, {})",
    }

    def uc(name: str, arg: str) -> str:
        return _uc[name].format(arg)

    def bc(name: str, lc: str, rc: str) -> str:
        return _bc[name].format(lc, rc)

    l_c    = uc(UNARY_NAMES[u_l],   FEATURES[feat_l])
    r_c    = uc(UNARY_NAMES[u_r],   FEATURES[feat_r])
    mid_c  = bc(BINARY_NAMES[bin_op], l_c, r_c)
    root_c = uc(UNARY_NAMES[u_root], mid_c)
    return (
        "def descriptor(rA, rB, rX, nA, nB, nX):\n"
        "    import numpy as np\n"
        f"    return float({root_c})\n"
    )


# ---------------------------------------------------------------------------
# RL controller — pure-NumPy REINFORCE
#
# Maintains one logit vector per tree-node position (same role as the
# controller's LSTM output in controller_PIDESv2.py).  The REINFORCE update
# equals: logits[i] += lr * advantage * ∂log_π/∂logits[i]
# where ∂log_π/∂logits[i] = one_hot(a) − softmax(logits[i]).
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class FEXController:
    """Independent categorical policy, one head per tree-node position."""

    def __init__(self, action_sizes: list[int], rng: np.random.Generator) -> None:
        self.logits = [np.zeros(n, dtype=np.float64) for n in action_sizes]
        self.rng    = rng

    def sample(self, greedy_eps: float = 0.0) -> tuple[list[int], list[np.ndarray]]:
        """Sample one action sequence; return (actions, log_prob_grads).

        log_prob_grads[i] = ∂log_π(a_i)/∂logits[i]
        """
        actions: list[int] = []
        grads:   list[np.ndarray] = []
        for logit in self.logits:
            probs = _softmax(logit)
            if self.rng.random() < greedy_eps:
                a = int(self.rng.integers(len(logit)))
            else:
                a = int(self.rng.choice(len(logit), p=probs))
            one_hot = np.zeros_like(probs)
            one_hot[a] = 1.0
            grads.append(one_hot - probs)      # ∂log π / ∂ logit
            actions.append(a)
        return actions, grads

    def update(
        self,
        batch_grads:     list[list[np.ndarray]],
        batch_advantages: list[float],
        lr: float,
    ) -> None:
        """REINFORCE gradient step."""
        n = len(batch_grads)
        for i, logit in enumerate(self.logits):
            g = sum(adv * bg[i] for adv, bg in zip(batch_advantages, batch_grads))
            logit += lr * g / n          # in-place update

    def get_probs(self) -> list[np.ndarray]:
        return [_softmax(l) for l in self.logits]


# ---------------------------------------------------------------------------
# Reward and final-metric helpers
# ---------------------------------------------------------------------------

def compute_reward(
    actions:      list[int],
    feature_vals: dict[str, np.ndarray],
    labels:       np.ndarray,
    train_mask:   np.ndarray,
    max_depth:    int,
) -> float:
    try:
        values = eval_expression(actions, feature_vals)
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        clf.fit(values[train_mask].reshape(-1, 1), labels[train_mask])
        preds = clf.predict(values[train_mask].reshape(-1, 1))
        return float((preds == labels[train_mask]).mean())
    except Exception:
        return 0.0


def compute_full_metrics(
    actions:      list[int],
    feature_vals: dict[str, np.ndarray],
    labels:       np.ndarray,
    train_mask:   np.ndarray,
    max_depth:    int,
) -> dict:
    values = eval_expression(actions, feature_vals)
    clf    = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(values[train_mask].reshape(-1, 1), labels[train_mask])
    tr = float((clf.predict(values[train_mask].reshape(-1, 1)) == labels[train_mask]).mean())
    te = float((clf.predict(values[~train_mask].reshape(-1, 1)) == labels[~train_mask]).mean())
    return {"train_accuracy": tr, "test_accuracy": te}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    controller:   FEXController,
    feature_vals: dict[str, np.ndarray],
    labels:       np.ndarray,
    train_mask:   np.ndarray,
    args:         argparse.Namespace,
) -> tuple[list[int], list[dict]]:
    baseline       = 0.0
    best_train_acc = 0.0
    best_actions: list[int] = []
    history: list[dict] = []

    for epoch in range(args.epochs):
        batch_rewards: list[float]            = []
        batch_grads:   list[list[np.ndarray]] = []
        batch_actions: list[list[int]]        = []

        for _ in range(args.batch_size):
            actions, grads = controller.sample(greedy_eps=args.greedy)
            reward = compute_reward(actions, feature_vals, labels, train_mask, args.max_depth)
            batch_rewards.append(reward)
            batch_grads.append(grads)
            batch_actions.append(actions)
            if reward > best_train_acc:
                best_train_acc = reward
                best_actions   = actions[:]

        mean_reward = float(np.mean(batch_rewards))
        baseline    = args.baseline_decay * baseline + (1.0 - args.baseline_decay) * mean_reward
        advantages  = [r - baseline for r in batch_rewards]

        controller.update(batch_grads, advantages, args.lr)

        record = {
            "epoch":          epoch,
            "mean_reward":    round(mean_reward, 6),
            "best_train_acc": round(best_train_acc, 6),
            "baseline":       round(baseline, 6),
        }
        history.append(record)

        if (epoch + 1) % max(1, args.epochs // 10) == 0:
            log.info(
                "Epoch %4d/%d | batch_mean=%.4f | baseline=%.4f | best_train=%.4f | %s",
                epoch + 1, args.epochs,
                mean_reward, baseline, best_train_acc,
                actions_to_expr(best_actions)[:60] if best_actions else "—",
            )

    return best_actions, history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FEX RL expression-tree search for perovskite stability"
    )
    parser.add_argument("--data_path",      default="perovskite-stability/TableS1.csv")
    parser.add_argument("--epochs",         type=int,   default=500,
                        help="REINFORCE epochs")
    parser.add_argument("--batch_size",     type=int,   default=32,
                        help="Candidate formulas sampled per epoch")
    parser.add_argument("--lr",             type=float, default=0.05,
                        help="Learning-rate for the RL controller logits")
    parser.add_argument("--greedy",         type=float, default=0.15,
                        help="ε-greedy exploration probability")
    parser.add_argument("--baseline_decay", type=float, default=0.9,
                        help="EMA decay for the REINFORCE baseline")
    parser.add_argument("--max_depth",      type=int,   default=2,
                        help="Max depth of the decision-tree classifier")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--output_dir",     default="search_runs/fex_results")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    # ------------------------------------------------------------------ data
    df           = load_dataset(args.data_path)
    labels       = df["exp_label"].values
    train_mask   = df["is_train"].values == 1
    feature_vals = {feat: df[feat].values for feat in FEATURES}

    log.info(
        "Loaded %d compounds  (train=%d  test=%d)",
        len(df), int(train_mask.sum()), int((~train_mask).sum()),
    )
    log.info(
        "Action space: %s  → %d total combinations",
        " × ".join(str(s) for s in ACTION_SIZES),
        int(np.prod(ACTION_SIZES)),
    )

    # -------------------------------------------------------- RL controller
    controller = FEXController(ACTION_SIZES, rng)

    log.info(
        "Starting FEX search: epochs=%d  batch=%d  greedy=%.2f  lr=%.4f",
        args.epochs, args.batch_size, args.greedy, args.lr,
    )

    best_actions, history = train(
        controller, feature_vals, labels, train_mask, args
    )

    if not best_actions:
        log.error("No valid formula found — all evaluations returned 0.")
        return

    # ------------------------------------------- final held-out evaluation
    metrics = compute_full_metrics(
        best_actions, feature_vals, labels, train_mask, args.max_depth
    )
    expr = actions_to_expr(best_actions)
    code = actions_to_code(best_actions)

    log.info("\n%s", "=" * 65)
    log.info("FEX  —  Best symbolic descriptor found")
    log.info("%s", "=" * 65)
    log.info("  Expression  : %s", expr)
    log.info("  Train acc   : %.4f  (%.1f%%)", metrics["train_accuracy"], metrics["train_accuracy"] * 100)
    log.info("  Test  acc   : %.4f  (%.1f%%)", metrics["test_accuracy"],  metrics["test_accuracy"]  * 100)
    log.info("  Actions     : %s", best_actions)
    log.info("%s", "=" * 65)

    # ---------------------------------------------------------------- save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "method":          "FEX",
        "best_expression": expr,
        "best_actions":    best_actions,
        "action_meaning": {
            "0_feat_left":   FEATURES[best_actions[0]],
            "1_left_unary":  UNARY_NAMES[best_actions[1]],
            "2_binary_op":   BINARY_NAMES[best_actions[2]],
            "3_feat_right":  FEATURES[best_actions[3]],
            "4_right_unary": UNARY_NAMES[best_actions[4]],
            "5_root_unary":  UNARY_NAMES[best_actions[5]],
        },
        "train_accuracy":  metrics["train_accuracy"],
        "test_accuracy":   metrics["test_accuracy"],
        "epochs":          args.epochs,
        "batch_size":      args.batch_size,
        "descriptor_code": code,
        "history":         history,
    }

    out_file = out_dir / "fex_results.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Results saved → %s", out_file)


if __name__ == "__main__":
    main()
