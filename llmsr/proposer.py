"""LLM-SR skeleton proposal with k-example context from buffer."""

import logging
from dataclasses import dataclass
from pathlib import Path

from llm_client import LLMClient
from llmsr.prompts import (
    IMPROVEMENT_SKELETON_TEMPLATE,
    INITIAL_SKELETON_TEMPLATE,
    PROBLEM_DESCRIPTION,
    SYSTEM_PROMPT,
)

log = logging.getLogger(__name__)

STRUCTURE_FIG = Path(__file__).parent / "assets" / "perovskite_structure_fig1.png"


@dataclass
class SkeletonProposal:
    function: str
    explanation: str
    formula: str


def _build_proposal(data: dict) -> SkeletonProposal:
    if "function" not in data:
        raise ValueError(f"LLM JSON missing required key 'function'; got: {list(data.keys())}")
    return SkeletonProposal(
        function=data["function"],
        explanation=data.get("explanation", ""),
        formula=data.get("formula", ""),
    )


def propose_initial_skeleton(client: LLMClient) -> SkeletonProposal:
    prompt = INITIAL_SKELETON_TEMPLATE.format(problem_desc=PROBLEM_DESCRIPTION)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    images = [STRUCTURE_FIG] if STRUCTURE_FIG.exists() else []
    data = client.query_json(messages, images=images if images else None)
    return _build_proposal(data)


def propose_skeleton_from_buffer(client: LLMClient, examples: list[dict]) -> SkeletonProposal:
    examples_text = "\n\n".join(
        f"#{i + 1} (accuracy {e['accuracy']:.1%}):\n```python\n{e['skeleton_code']}\n```\nExplanation: {e['explanation']}"
        for i, e in enumerate(examples)
    )
    prompt = IMPROVEMENT_SKELETON_TEMPLATE.format(
        problem_desc=PROBLEM_DESCRIPTION,
        k=len(examples),
        examples=examples_text,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    images = [STRUCTURE_FIG] if STRUCTURE_FIG.exists() else []
    data = client.query_json(messages, images=images if images else None)
    return _build_proposal(data)
