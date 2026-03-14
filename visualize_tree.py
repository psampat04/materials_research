"""Generate a Graphviz tree diagram from a search_state.json file."""

import json
import sys
import textwrap
from pathlib import Path

try:
    import graphviz
except ImportError:
    sys.exit("Install graphviz: pip install graphviz")


def truncate(text: str, max_len: int = 60) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "…"


def accuracy_color(test_acc: float) -> str:
    if test_acc >= 0.80:
        return "#2ecc71"
    if test_acc >= 0.75:
        return "#a8e6cf"
    if test_acc >= 0.70:
        return "#ffeaa7"
    if test_acc >= 0.65:
        return "#fdcb6e"
    return "#e17055"


def build_tree(state_path: str, output_path: str | None = None):
    state_path = Path(state_path)
    with open(state_path) as f:
        data = json.load(f)

    nodes = data["nodes"]
    root_children = data["root_children"]

    if output_path is None:
        output_path = str(state_path.parent / "mcts_tree")

    dot = graphviz.Digraph(
        "MCTS Tree",
        format="png",
        graph_attr={
            "rankdir": "TB",
            "label": f"MCTS Descriptor Search Tree  ({state_path.parent.name})\nColor by test accuracy",
            "labelloc": "t",
            "fontsize": "18",
            "nodesep": "0.4",
            "ranksep": "0.8",
        },
        node_attr={"shape": "box", "style": "filled,rounded", "fontsize": "10"},
        edge_attr={"fontsize": "9"},
    )

    dot.node("ROOT", "ROOT\n(virtual)", shape="ellipse", style="filled", fillcolor="#d5d8dc")

    for nid, node in nodes.items():
        train_acc = node["metrics"].get("train_accuracy", 0)
        test_acc = node["metrics"].get("test_accuracy", 0)
        formula = node.get("formula", "")
        formula_short = truncate(formula, 55) if formula else "(no formula)"

        label = (
            f"{nid}  (depth {node['depth']})\n"
            f"{formula_short}\n"
            f"Train: {train_acc:.1%}  |  Test: {test_acc:.1%}"
        )

        color = accuracy_color(test_acc)
        dot.node(nid, label, fillcolor=color)

    for rid in root_children:
        node = nodes[rid]
        test_acc = node["metrics"].get("test_accuracy", 0)
        dot.edge("ROOT", rid)

    for nid, node in nodes.items():
        for cid in node.get("children_ids", []):
            child = nodes[cid]
            parent_test = node["metrics"].get("test_accuracy", 0)
            child_test = child["metrics"].get("test_accuracy", 0)
            delta = child_test - parent_test
            sign = "+" if delta >= 0 else ""
            dot.edge(nid, cid, label=f"{sign}{delta:.1%}")

    dot.render(output_path, cleanup=True)
    print(f"Tree saved to {output_path}.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_tree.py <search_state.json> [output_path]")
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) > 2 else None
    build_tree(sys.argv[1], out)
