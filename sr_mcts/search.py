"""SR-MCTS: token-level MCTS with PUCT and LLM-guided priors.

A third search variant alongside `mcts` and `llmsr`. Based on Kadam (2024),
"GPT-Guided Monte Carlo Tree Search for Symbolic Regression"
(arXiv:2411.04459).

Conceptual difference from the other two modes:
  * `mcts`  — MCTS over whole-formula refinements; LLM rewrites entire
              formulas per node.
  * `llmsr` — Experience-buffer loop; LLM proposes new formula *skeletons*,
              parameters are fit numerically.
  * `sr_mcts` (this file) — Each MCTS node is a *partial* prefix expression;
              the LLM acts as a policy that scores the next single token.
              PUCT selects promising partial expressions; rollouts complete
              them uniformly. When a new best complete formula is found,
              it is dropped through the shared `evaluate_candidate`
              pipeline so the result lives in the same `SearchState`
              as MCTS/LLMSR results.
"""

import logging
import math
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from evaluator import evaluate_candidate
from llm_client import LLMClient
from sr_mcts.grammar import (
    VARS,
    State,
    apply_action,
    initial_state,
    legal_actions,
    tokens_to_code,
    tokens_to_pretty,
    wrap_as_descriptor,
)
from sr_mcts.policy import LLMPolicy, uniform_policy
from state import FormulaNode, SearchState

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MCTS primitives
# ---------------------------------------------------------------------------

class Node:
    """One partial expression in the search tree."""

    __slots__ = ("state", "parent", "action_from_parent",
                 "children", "priors", "expanded", "N", "W")

    def __init__(self, state: State, parent=None, action_from_parent=None) -> None:
        self.state              = state
        self.parent             = parent
        self.action_from_parent = action_from_parent
        self.children: dict[str, "Node"] = {}
        self.priors:  dict[str, float]  = {}
        self.expanded = False
        self.N        = 0
        self.W        = 0.0

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


def puct_select(parent: Node, c_puct: float) -> tuple[str, Node]:
    """PUCT:  argmax_a  Q(a) + c · P(a) · √N(parent) / (1 + N_child(a))."""
    sqrt_n = math.sqrt(parent.N) if parent.N > 0 else 1.0
    best_action, best_score = None, -float("inf")
    for action, child in parent.children.items():
        prior = parent.priors.get(action, 1.0 / len(parent.children))
        u = c_puct * prior * sqrt_n / (1 + child.N)
        score = child.Q + u
        if score > best_score:
            best_score = score
            best_action = action
    return best_action, parent.children[best_action]


def expand(node: Node, policy, max_len: int) -> None:
    actions = legal_actions(node.state, max_len)
    if not actions:
        node.expanded = True
        return
    node.priors = policy(node.state, actions)
    for a in actions:
        node.children[a] = Node(
            apply_action(node.state, a),
            parent=node,
            action_from_parent=a,
        )
    node.expanded = True


def rollout(start: State, policy, max_len: int, rng: np.random.Generator) -> tuple | None:
    s = start
    while not s.is_terminal:
        actions = legal_actions(s, max_len)
        if not actions:
            return None
        p = policy(s, actions)
        probs = np.array([p[a] for a in actions], dtype=float)
        probs = probs / probs.sum()
        idx = int(rng.choice(len(actions), p=probs))
        s = apply_action(s, actions[idx])
    return s.tokens


def mcts_iteration(
    root: Node,
    expand_policy,
    rollout_policy,
    max_len: int,
    reward_fn,
    rng: np.random.Generator,
    c_puct: float,
) -> tuple[tuple | None, float]:
    """One SELECT → EXPAND → ROLLOUT → BACKUP pass."""
    path = [root]
    node = root
    while node.expanded and node.children:
        _, node = puct_select(node, c_puct)
        path.append(node)

    if not node.state.is_terminal and not node.expanded:
        expand(node, expand_policy, max_len)

    if node.state.is_terminal:
        tokens = node.state.tokens
    elif node.children:
        actions = list(node.children.keys())
        probs = np.array([node.priors[a] for a in actions], dtype=float)
        probs = probs / probs.sum()
        a = actions[int(rng.choice(len(actions), p=probs))]
        child = node.children[a]
        path.append(child)
        tokens = rollout(child.state, rollout_policy, max_len, rng)
    else:
        tokens = None

    value = reward_fn(tokens) if tokens else 0.0

    for n in path:
        n.N += 1
        n.W += value
    return tokens, value


# ---------------------------------------------------------------------------
# Fast in-loop reward (training-set accuracy of a depth-k DT classifier)
# ---------------------------------------------------------------------------

def _make_reward_fn(df: pd.DataFrame, max_depth: int, train_split_label: int):
    labels       = df["exp_label"].values
    train_mask   = df["is_train"].values == train_split_label
    feature_vals = {v: df[v].values.astype(float) for v in VARS}

    def reward(tokens):
        if not tokens:
            return 0.0
        expr = tokens_to_code(tokens)
        try:
            values = eval(expr, {"np": np}, feature_vals)  # noqa: S307
            values = np.asarray(values, dtype=float)
            if values.ndim == 0 or not np.isfinite(values).all():
                return 0.0
            clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            clf.fit(values[train_mask].reshape(-1, 1), labels[train_mask])
            preds = clf.predict(values[train_mask].reshape(-1, 1))
            return float((preds == labels[train_mask]).mean())
        except Exception:
            return 0.0

    return reward


# ---------------------------------------------------------------------------
# Top-level entry — called from run_search.py when cfg.search.mode == "sr_mcts"
# ---------------------------------------------------------------------------

def run_sr_mcts(
    client: LLMClient,
    state: SearchState,
    df: pd.DataFrame,
    plot_dir: Path,
    cfg,
    *,
    state_save_path: Path,
) -> SearchState:
    """Main SR-MCTS loop.

    Budget is interpreted as the number of MCTS iterations (consistent with
    how LLMSR and MCTS use `cfg.mcts.budget`). Each newly discovered best
    expression is pushed through the shared evaluator and added to
    `state` as a FormulaNode, keeping outputs compatible with the other
    two modes.
    """
    plot_dir.mkdir(parents=True, exist_ok=True)

    budget         = cfg.mcts.budget
    max_len        = int(getattr(cfg.sr_mcts, "max_len", 11))
    c_puct         = float(getattr(cfg.sr_mcts, "c_puct", 1.5))
    seed           = int(getattr(cfg.sr_mcts, "seed", 42))
    use_llm_prior  = bool(getattr(cfg.sr_mcts, "use_llm_prior", True))

    rng = np.random.default_rng(seed)

    # ---- policies --------------------------------------------------------
    rollout_policy = uniform_policy
    llm_policy: LLMPolicy | None = None
    if use_llm_prior:
        llm_policy = LLMPolicy(client)
        expand_policy = llm_policy
        log.info("SR-MCTS: LLM-guided prior at expansion, uniform rollouts.")
    else:
        expand_policy = uniform_policy
        log.info("SR-MCTS: uniform prior (no LLM).")

    # ---- reward ----------------------------------------------------------
    reward_fn = _make_reward_fn(
        df,
        max_depth=cfg.eval.decision_tree_max_depth,
        train_split_label=cfg.eval.train_split_label,
    )

    # ---- resume: seed best_reward from any previously stored nodes --------
    best_reward = max((n.accuracy for n in state.nodes.values()), default=0.0)

    root = Node(initial_state())
    expand(root, expand_policy, max_len)

    log.info(
        "SR-MCTS starting: budget=%d  max_len=%d  c_puct=%.2f  best_so_far=%.3f",
        budget, max_len, c_puct, best_reward,
    )

    it_start = state.budget_used
    for it in range(it_start, budget):
        tokens, value = mcts_iteration(
            root, expand_policy, rollout_policy,
            max_len, reward_fn, rng, c_puct,
        )
        state.budget_used = it + 1

        # Promote strictly better complete formulas into the shared state so
        # they get a plot + full metrics (train/test/CV) and show up in the
        # usual ranked output alongside LLMSR and MCTS results.
        if tokens and value > best_reward + 1e-9:
            best_reward = value
            _promote_to_state(
                tokens=tokens,
                client_or_none=llm_policy,
                state=state,
                df=df,
                plot_dir=plot_dir,
                cfg=cfg,
                iteration=it + 1,
            )
            log.info(
                "Iter %5d | NEW BEST train_acc=%.4f | expr=%s",
                it + 1, value, tokens_to_pretty(tokens),
            )

        if (it + 1) % max(1, budget // 10) == 0:
            extra = ""
            if llm_policy is not None:
                extra = f" | llm_calls={llm_policy.calls} cache={len(llm_policy.cache)}"
            log.info(
                "Iter %5d/%d | best train_acc=%.4f%s",
                it + 1, budget, best_reward, extra,
            )
            state.save(state_save_path)

    # Final save
    state.save(state_save_path)
    log.info("SR-MCTS done. Best train accuracy: %.4f", best_reward)
    return state


def _promote_to_state(
    *,
    tokens: tuple,
    client_or_none,
    state: SearchState,
    df: pd.DataFrame,
    plot_dir: Path,
    cfg,
    iteration: int,
) -> None:
    """Run the shared evaluator on a newly-best formula and append a node."""
    code = wrap_as_descriptor(tokens_to_code(tokens))
    node_id = uuid.uuid4().hex[:8]
    result = evaluate_candidate(
        code,
        df,
        plot_dir,
        node_id,
        decision_tree_max_depth=cfg.eval.decision_tree_max_depth,
        train_split_label=cfg.eval.train_split_label,
        cv_folds=getattr(cfg.eval, "cv_folds", 5),
    )
    if result.error:
        log.warning("Evaluator rejected SR-MCTS formula %s: %s", node_id, result.error)
        return

    formula = tokens_to_pretty(tokens)
    node = FormulaNode(
        id=node_id,
        parent_id=None,
        code=code,
        description=f"SR-MCTS token-level PUCT (max_len={cfg.sr_mcts.max_len})",
        formula=formula,
        accuracy=result.accuracy,
        metrics={
            "train_accuracy":      result.train_accuracy,
            "test_accuracy":       result.test_accuracy,
            "cv_accuracy":         result.cv_accuracy,
            "cv_std":              result.cv_std,
            "cv_fold_scores":      result.cv_fold_scores,
            "false_positive_rate": result.false_positive_rate,
            "per_anion_accuracy":  result.per_anion_accuracy,
            "metrics_summary":     result.metrics_summary,
            "tokens":              list(tokens),
            "mode":                "sr_mcts",
        },
        plot_path=result.plot_path,
        visit_count=1,
        total_reward=result.accuracy,
        children_ids=[],
        depth=iteration,
    )
    state.add_node(node)
