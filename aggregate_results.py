"""Aggregate all formulas across search runs and rank by accuracy."""

import json
from pathlib import Path

SEARCH_RUNS = Path(__file__).parent / "search_runs"
OUTPUT = Path(__file__).parent / "ranked_formulas.json"


def main():
    all_formulas = []

    for state_file in sorted(SEARCH_RUNS.glob("*/search_state.json")):
        run_name = state_file.parent.name
        if state_file.stat().st_size == 0:
            print(f"Skipping empty file: {state_file}")
            continue
        with open(state_file) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON: {state_file} ({e})")
                continue

        for node in data["nodes"].values():
            all_formulas.append({
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
            })

    all_formulas.sort(key=lambda x: x["accuracy"], reverse=True)

    for rank, entry in enumerate(all_formulas, 1):
        entry["rank"] = rank

    output = {
        "total_formulas": len(all_formulas),
        "total_runs": len(list(SEARCH_RUNS.glob("*/search_state.json"))),
        "formulas": all_formulas,
    }

    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Aggregated {len(all_formulas)} formulas from {output['total_runs']} runs")
    print(f"Saved to {OUTPUT}")
    print(f"\nTop 10:")
    for entry in all_formulas[:10]:
        print(f"  #{entry['rank']} | {entry['accuracy']:.1%} (train={entry['train_accuracy']:.1%}, test={entry['test_accuracy']:.1%}) | run={entry['run']} node={entry['node_id']} depth={entry['depth']}")
        print(f"       {entry['formula'][:80]}")


if __name__ == "__main__":
    main()
