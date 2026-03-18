"""Monte Carlo Tree Search over descriptor formulas."""

import logging
import math
import uuid
from pathlib import Path

import pandas as pd

from debugger import debug_function
from evaluator import evaluate_candidate
from llm_client import LLMClient
from proposer import propose_improvement, propose_initial
from state import FormulaNode, SearchState

log = logging.getLogger(__name__)


def _generate_node_id() -> str:
    return uuid.uuid4().hex[:8]


def _ucb1(node: FormulaNode, parent_visits: int, c: float) -> float:
    if node.visit_count == 0:
        return float("inf")
    exploitation = node.total_reward / node.visit_count
    exploration = c * math.sqrt(math.log(parent_visits) / node.visit_count)
    return exploitation + exploration


def select_node(state: SearchState, cfg) -> FormulaNode | None:
    """UCB1 selection: traverse from root children to a leaf or expandable node."""
    if not state.root_children:
        return None

    ucb_constant = cfg.mcts.ucb_constant
    max_depth = cfg.mcts.max_depth
    # Allow multiple children per node so the search forms a real tree
    max_children = getattr(cfg.mcts, "max_children_per_node", 3)

    total_root_visits = sum(state.nodes[cid].visit_count for cid in state.root_children)
    if total_root_visits == 0:
        total_root_visits = 1

    best_id = max(
        state.root_children,
        key=lambda cid: _ucb1(state.nodes[cid], total_root_visits, ucb_constant),
    )
    current = state.nodes[best_id]

    # Traverse down using UCB, but stop at any node that is not "fully expanded"
    # so that each node can spawn multiple children instead of a single chain.
    while current.depth < max_depth:
        if len(current.children_ids) < max_children:
            # This node can still create new children — expand from here.
            break

        # All children already exist; select the best child and continue.
        parent_visits = current.visit_count if current.visit_count > 0 else 1
        best_child_id = max(
            current.children_ids,
            key=lambda cid: _ucb1(state.nodes[cid], parent_visits, ucb_constant),
        )
        current = state.nodes[best_child_id]

    return current


def _try_evaluate(
    code: str,
    client: LLMClient,
    df: pd.DataFrame,
    plot_dir: Path,
    node_id: str,
    state: SearchState,
    cfg,
):
    """Try to evaluate a function, with one debug attempt on failure."""
    cv_folds = getattr(cfg.eval, "cv_folds", 5)
    result = evaluate_candidate(
        code,
        df,
        plot_dir,
        node_id,
        decision_tree_max_depth=cfg.eval.decision_tree_max_depth,
        train_split_label=cfg.eval.train_split_label,
        cv_folds=cv_folds,
    )

    if not result.error:
        return result, code

    log.warning("Evaluation failed: %s — attempting debug fix", result.error)
    fixed_code = debug_function(client, code, result.error)
    state.debug_calls += 1

    result2 = evaluate_candidate(
        fixed_code,
        df,
        plot_dir,
        node_id,
        decision_tree_max_depth=cfg.eval.decision_tree_max_depth,
        train_split_label=cfg.eval.train_split_label,
        cv_folds=cv_folds,
    )
    if not result2.error:
        return result2, fixed_code
    return result2, code


def _result_to_node(
    node_id: str, parent_id: str | None, code: str, proposal, result, depth: int
) -> FormulaNode:
    return FormulaNode(
        id=node_id,
        parent_id=parent_id,
        code=code,
        description=proposal.explanation,
        formula=proposal.formula,
        accuracy=result.accuracy,
        metrics={
            "train_accuracy": result.train_accuracy,
            "test_accuracy": result.test_accuracy,
            "cv_accuracy": result.cv_accuracy,
            "cv_std": result.cv_std,
            "cv_fold_scores": result.cv_fold_scores,
            "false_positive_rate": result.false_positive_rate,
            "per_anion_accuracy": result.per_anion_accuracy,
            "metrics_summary": result.metrics_summary,
        },
        plot_path=result.plot_path,
        visit_count=1,
        total_reward=0.0,
        children_ids=[],
        depth=depth,
    )


def expand_initial(
    client: LLMClient, state: SearchState, df: pd.DataFrame, plot_dir: Path, cfg
) -> FormulaNode | None:
    """Propose and evaluate an initial formula (root child)."""
    proposal = propose_initial(client)
    state.total_llm_calls += 1
    node_id = _generate_node_id()

    result, final_code = _try_evaluate(proposal.function, client, df, plot_dir, node_id, state, cfg)

    if result.error:
        log.warning("Initial formula failed even after debugging: %s", result.error)
        return None

    node = _result_to_node(node_id, None, final_code, proposal, result, depth=0)
    state.add_node(node)
    state.budget_used += 1
    return node


def expand_child(
    parent: FormulaNode,
    client: LLMClient,
    state: SearchState,
    df: pd.DataFrame,
    plot_dir: Path,
    cfg,
) -> FormulaNode | None:
    """Propose an improvement of a parent formula and evaluate it."""
    if parent.depth >= cfg.mcts.max_depth:
        log.info("Max depth reached at node %s", parent.id)
        return None

    proposal = propose_improvement(
        client,
        parent.code,
        parent.formula,
        parent.description,
        parent.metrics["metrics_summary"],
        Path(parent.plot_path),
    )
    state.total_llm_calls += 1
    node_id = _generate_node_id()

    result, final_code = _try_evaluate(proposal.function, client, df, plot_dir, node_id, state, cfg)

    if result.error:
        log.warning("Improved formula failed even after debugging: %s", result.error)
        return None

    node = _result_to_node(node_id, parent.id, final_code, proposal, result, depth=parent.depth + 1)
    state.add_node(node)
    state.budget_used += 1
    return node


def backpropagate(state: SearchState, node: FormulaNode) -> None:
    """Recompute all rewards and backpropagate up the tree."""
    rewards = state.recompute_ranks()

    for nid, reward in rewards.items():
        n = state.nodes[nid]
        n.total_reward = reward * n.visit_count

    current_id = node.parent_id
    while current_id is not None:
        parent = state.nodes[current_id]
        parent.visit_count += 1
        parent.total_reward += rewards[node.id]
        current_id = parent.parent_id


def run_mcts(
    client: LLMClient,
    state: SearchState,
    df: pd.DataFrame,
    plot_dir: Path,
    cfg,
    *,
    state_save_path: Path,
) -> SearchState:
    """Main MCTS loop."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    budget = cfg.mcts.budget
    initial_samples = cfg.mcts.initial_samples

    while len(state.root_children) < initial_samples and state.budget_used < budget:
        log.info(
            "=== Initial sample %d/%d (budget %d/%d) ===",
            len(state.root_children) + 1,
            initial_samples,
            state.budget_used,
            budget,
        )
        node = expand_initial(client, state, df, plot_dir, cfg)
        if node:
            backpropagate(state, node)
            log.info("Initial node %s: accuracy=%.3f", node.id, node.accuracy)
        state.save(state_save_path)

    while state.budget_used < budget:
        log.info("=== MCTS iteration (budget %d/%d) ===", state.budget_used, budget)

        selected = select_node(state, cfg)
        if selected is None:
            log.warning("No node selected, stopping")
            break

        log.info(
            "Selected node %s (depth=%d, accuracy=%.3f)",
            selected.id,
            selected.depth,
            selected.accuracy,
        )

        child = expand_child(selected, client, state, df, plot_dir, cfg)
        if child:
            backpropagate(state, child)
            log.info(
                "New node %s: accuracy=%.3f (parent %s: %.3f)",
                child.id,
                child.accuracy,
                selected.id,
                selected.accuracy,
            )
        else:
            log.info(
                "Selected node %s (depth=%d) could not be expanded (likely max depth); "
                "stopping search.",
                selected.id,
                selected.depth,
            )
            continue

        state.save(state_save_path)

    return state
