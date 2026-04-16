"""LLM-SR search loop: experience buffer + skeleton parameter optimisation."""

import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from debugger import debug_function
from evaluator import evaluate_candidate
from llm_client import LLMClient
from llmsr.optimizer import bake_params, count_params, fit_params
from llmsr.proposer import propose_initial_skeleton, propose_skeleton_from_buffer
from state import FormulaNode, SearchState

log = logging.getLogger(__name__)

_BUFFER_SIZE = 50
_SAMPLE_K = 3
_BOLTZMANN_TEMP = 0.3


def _sample_from_buffer(buffer: list[dict], k: int) -> list[dict]:
    """Boltzmann-weighted sample of k entries — biased toward high accuracy but not greedy."""
    if len(buffer) <= k:
        return list(buffer)
    scores = np.array([b["accuracy"] for b in buffer])
    # Shift by max for numerical stability
    weights = np.exp((scores - scores.max()) / _BOLTZMANN_TEMP)
    weights /= weights.sum()
    indices = np.random.choice(len(buffer), size=k, replace=False, p=weights)
    return [buffer[i] for i in indices]


def _to_node(
    node_id: str,
    skeleton_code: str,
    baked_code: str,
    proposal,
    result,
    params: np.ndarray,
    iteration: int,
) -> FormulaNode:
    return FormulaNode(
        id=node_id,
        parent_id=None,
        code=baked_code,
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
            "params": [float(p) for p in params],
            "skeleton_code": skeleton_code,
        },
        plot_path=result.plot_path,
        visit_count=1,
        total_reward=result.accuracy,
        children_ids=[],
        depth=iteration,
    )


def run_llmsr(
    client: LLMClient,
    state: SearchState,
    df: pd.DataFrame,
    plot_dir: Path,
    cfg,
    *,
    state_save_path: Path,
) -> SearchState:
    """Main LLM-SR loop."""
    plot_dir.mkdir(parents=True, exist_ok=True)

    budget = cfg.mcts.budget
    initial_samples = cfg.mcts.initial_samples
    cv_folds = getattr(cfg.eval, "cv_folds", 5)

    # Seed buffer from any previously saved nodes (resume support)
    buffer: list[dict] = [
        {
            "accuracy": n.accuracy,
            "skeleton_code": n.metrics.get("skeleton_code", n.code),
            "explanation": n.description,
        }
        for n in sorted(state.nodes.values(), key=lambda x: x.accuracy, reverse=True)
        if n.accuracy > 0
    ]

    iteration = state.budget_used

    while state.budget_used < budget:
        log.info("=== LLM-SR iteration %d/%d ===", state.budget_used + 1, budget)

        # --- Propose skeleton ---
        if len(buffer) < initial_samples:
            proposal = propose_initial_skeleton(client)
        else:
            examples = _sample_from_buffer(buffer, _SAMPLE_K)
            proposal = propose_skeleton_from_buffer(client, examples)
        state.total_llm_calls += 1

        skeleton_code = proposal.function
        node_id = uuid.uuid4().hex[:8]

        # --- Fit params ---
        n_params = count_params(skeleton_code)
        try:
            if n_params > 0:
                log.info("Fitting %d params for node %s", n_params, node_id)
                params, _ = fit_params(skeleton_code, df, n_params)
                baked_code = bake_params(skeleton_code, params)
            else:
                params = np.array([])
                baked_code = skeleton_code
        except ValueError as e:
            log.warning("Skipping node %s — param fitting failed: %s", node_id, e)
            state.budget_used += 1
            iteration += 1
            state.save(state_save_path)
            continue

        # --- Evaluate ---
        result = evaluate_candidate(
            baked_code,
            df,
            plot_dir,
            node_id,
            decision_tree_max_depth=cfg.eval.decision_tree_max_depth,
            train_split_label=cfg.eval.train_split_label,
            cv_folds=cv_folds,
        )

        if result.error:
            log.warning("Evaluation failed: %s — attempting debug fix", result.error)
            fixed_code = debug_function(client, baked_code, result.error)
            state.debug_calls += 1
            result = evaluate_candidate(
                fixed_code,
                df,
                plot_dir,
                node_id,
                decision_tree_max_depth=cfg.eval.decision_tree_max_depth,
                train_split_label=cfg.eval.train_split_label,
                cv_folds=cv_folds,
            )
            if not result.error:
                baked_code = fixed_code

        # --- Store ---
        if not result.error:
            node = _to_node(node_id, skeleton_code, baked_code, proposal, result, params, iteration)
            state.add_node(node)

            buffer.append({
                "accuracy": result.accuracy,
                "skeleton_code": skeleton_code,
                "explanation": proposal.explanation,
            })
            buffer.sort(key=lambda x: x["accuracy"], reverse=True)
            buffer = buffer[:_BUFFER_SIZE]

            log.info(
                "Node %s: accuracy=%.3f | params=%s | buffer_size=%d",
                node_id, result.accuracy, [round(p, 4) for p in params], len(buffer),
            )
        else:
            log.warning("Node %s failed after debug attempt: %s", node_id, result.error)

        state.budget_used += 1
        iteration += 1
        state.save(state_save_path)

    return state
