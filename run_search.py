"""Entry point for the MCTS formula search agent."""

import logging
import os
from datetime import datetime
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from evaluator import load_dataset
from llm_client import LLMClient
from llmsr import run_llmsr
from mcts import run_mcts
from sr_mcts import run_sr_mcts
from state import SearchState

log = logging.getLogger(__name__)


def _print_top_formulas(state: SearchState, k: int = 10) -> None:
    top = state.top_k(k)
    print("\n" + "=" * 70)
    print(f"Top {min(k, len(top))} formulas by accuracy")
    print("=" * 70)
    for i, node in enumerate(top, 1):
        print(f"\n--- #{i} (node {node.id}, depth={node.depth}) ---")
        print(f"  Accuracy: {node.accuracy:.1%}")
        per_anion = node.metrics["per_anion_accuracy"]
        if per_anion:
            anion_str = ", ".join(f"{a}={v:.0%}" for a, v in per_anion.items())
            print(f"  Per-anion: {anion_str}")
        if node.formula:
            print(f"  Formula: {node.formula}")
        if node.description:
            print(f"  Explanation: {node.description}")
        print(f"  Code:\n{_indent(node.code, 4)}")


def _indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.split("\n"))


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in your .env file")

    orig_cwd = hydra.utils.get_original_cwd()
    data_path = Path(orig_cwd) / cfg.eval.data_path

    mode = getattr(cfg.search, "mode", "mcts")
    run_dir = Path(orig_cwd) / cfg.search.state_path / mode
    if cfg.search.resume:
        state_file = Path(orig_cwd) / cfg.search.resume
        state = SearchState.load(state_file)
        run_dir = state_file.parent
    else:
        run_dir = run_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        state = SearchState()

    plot_dir = run_dir / "plots"
    state_save_path = run_dir / "search_state.json"

    log.info("Run directory: %s", run_dir)
    log.info("Budget: %d, Initial samples: %d", cfg.mcts.budget, cfg.mcts.initial_samples)

    client = LLMClient(cfg, api_key)

    df = load_dataset(str(data_path))
    log.info("Loaded %d compounds from %s", len(df), data_path)

    if mode == "llmsr":
        state = run_llmsr(client, state, df, plot_dir, cfg, state_save_path=state_save_path)
    elif mode == "sr_mcts":
        state = run_sr_mcts(client, state, df, plot_dir, cfg, state_save_path=state_save_path)
    else:
        state = run_mcts(client, state, df, plot_dir, cfg, state_save_path=state_save_path)

    _print_top_formulas(state)

    print(f"\nSearch complete. Budget used: {state.budget_used}/{cfg.mcts.budget}")
    print(f"Total LLM calls: {state.total_llm_calls} (debug: {state.debug_calls})")
    print(f"LLM usage: {client.usage_summary()}")
    print(f"State saved to: {state_save_path}")


if __name__ == "__main__":
    main()
