"""Aggregate all formulas across search runs and rank by accuracy."""

import json
from pathlib import Path

SEARCH_RUNS = Path(__file__).parent / "search_runs"
OUTPUT_DIR = Path(__file__).parent / "ranked_results"
OUTPUT_MCTS = OUTPUT_DIR / "ranked_formulas_mcts.json"
OUTPUT_LLMSR = OUTPUT_DIR / "ranked_formulas_llmsr.json"
OUTPUT_SR_MCTS = OUTPUT_DIR / "ranked_formulas_sr_mcts.json"
OUTPUT_ALL = OUTPUT_DIR / "ranked_formulas.json"
OUTPUT_DIR.mkdir(exist_ok=True)


def _load_run(state_file: Path, algorithm: str) -> list[dict]:
    if state_file.stat().st_size == 0:
        print(f"Skipping empty file: {state_file}")
        return []
    with open(state_file) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Skipping malformed JSON: {state_file} ({e})")
            return []

    run_name = state_file.parent.name
    entries = []
    for node in data["nodes"].values():
        entry = {
            "algorithm": algorithm,
            "run": run_name,
            "node_id": node["id"],
            "depth": node["depth"],
            "parent_id": node["parent_id"],
            "accuracy": node["accuracy"],
            "train_accuracy": node["metrics"].get("train_accuracy"),
            "test_accuracy": node["metrics"].get("test_accuracy"),
            "per_anion_accuracy": node["metrics"].get("per_anion_accuracy", {}),
            "formula": node["formula"],
            "explanation": node["description"],
            "code": node["code"],
        }
        # LLM-SR specific fields
        if algorithm == "llmsr":
            entry["skeleton_code"] = node["metrics"].get("skeleton_code", "")
            entry["params"] = node["metrics"].get("params", [])
        # SR-MCTS specific fields (prefix tokens)
        if algorithm == "sr_mcts":
            entry["tokens"] = node["metrics"].get("tokens", [])
        entries.append(entry)
    return entries


def _write_output(formulas: list[dict], output_path: Path, run_count: int) -> None:
    formulas.sort(key=lambda x: x["accuracy"], reverse=True)
    for rank, entry in enumerate(formulas, 1):
        entry["rank"] = rank

    output = {
        "total_formulas": len(formulas),
        "total_runs": run_count,
        "formulas": formulas,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(formulas)} formulas from {run_count} runs → {output_path.name}")
    print("Top 5:")
    for entry in formulas[:5]:
        anion = entry.get("per_anion_accuracy", {})
        anion_str = " ".join(f"{a}={v:.0%}" for a, v in anion.items()) if anion else "N/A"
        print(f"  #{entry['rank']} | {entry['accuracy']:.1%} | run={entry['run']} | {anion_str}")
        if entry.get("formula"):
            print(f"       {entry['formula'][:80]}")
    print()


def main():
    mcts_formulas, llmsr_formulas, sr_mcts_formulas = [], [], []
    mcts_runs, llmsr_runs, sr_mcts_runs = 0, 0, 0

    for state_file in sorted((SEARCH_RUNS / "mcts").glob("*/search_state.json")):
        mcts_formulas.extend(_load_run(state_file, "mcts"))
        mcts_runs += 1

    for state_file in sorted((SEARCH_RUNS / "llmsr").glob("*/search_state.json")):
        llmsr_formulas.extend(_load_run(state_file, "llmsr"))
        llmsr_runs += 1

    for state_file in sorted((SEARCH_RUNS / "sr_mcts").glob("*/search_state.json")):
        sr_mcts_formulas.extend(_load_run(state_file, "sr_mcts"))
        sr_mcts_runs += 1

    if mcts_formulas:
        _write_output(mcts_formulas, OUTPUT_MCTS, mcts_runs)
    else:
        print("No MCTS runs found.")

    if llmsr_formulas:
        _write_output(llmsr_formulas, OUTPUT_LLMSR, llmsr_runs)
    else:
        print("No LLM-SR runs found.")

    if sr_mcts_formulas:
        _write_output(sr_mcts_formulas, OUTPUT_SR_MCTS, sr_mcts_runs)
    else:
        print("No SR-MCTS runs found.")

    # Combined file with all formulas
    all_formulas = mcts_formulas + llmsr_formulas + sr_mcts_formulas
    if all_formulas:
        _write_output(all_formulas, OUTPUT_ALL, mcts_runs + llmsr_runs + sr_mcts_runs)


if __name__ == "__main__":
    main()
