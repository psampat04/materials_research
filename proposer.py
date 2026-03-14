"""LLM prompt construction for proposing and improving descriptor formulas."""

import logging
from dataclasses import dataclass
from pathlib import Path

from llm_client import LLMClient
from prompts import (
    IMPROVEMENT_PROMPT_TEMPLATE,
    INITIAL_PROMPT_TEMPLATE,
    PROBLEM_DESCRIPTION,
    SYSTEM_PROMPT,
)

log = logging.getLogger(__name__)

STRUCTURE_FIG = Path(__file__).parent / "assets" / "perovskite_structure_fig1.png"


@dataclass
class Proposal:
    function: str
    explanation: str
    formula: str


def _build_proposal(data: dict) -> Proposal:
    missing = [k for k in ("function", "explanation", "formula") if k not in data]
    if missing:
        raise ValueError(f"LLM JSON missing keys: {missing}; got: {list(data.keys())}")
    return Proposal(
        function=data["function"],
        explanation=data["explanation"],
        formula=data["formula"],
    )


def propose_initial(client: LLMClient) -> Proposal:
    prompt = INITIAL_PROMPT_TEMPLATE.format(problem_desc=PROBLEM_DESCRIPTION)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    data = client.query_json(messages) # , images=[STRUCTURE_FIG] cannot interpret images right now
    return _build_proposal(data)


def propose_improvement(
    client: LLMClient,
    parent_code: str,
    parent_formula: str,
    parent_explanation: str,
    metrics_summary: str,
    plot_image: Path,
) -> Proposal:
    prompt = IMPROVEMENT_PROMPT_TEMPLATE.format(
        problem_desc=PROBLEM_DESCRIPTION,
        parent_code=parent_code,
        parent_formula=parent_formula,
        parent_explanation=parent_explanation,
        metrics_summary=metrics_summary,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    # images = [STRUCTURE_FIG]
    images = []
    if STRUCTURE_FIG.exists():
        images.append(STRUCTURE_FIG)
    if plot_image.exists():
        images.append(plot_image)
    data = client.query_json(messages, images=images if images else None)
    return _build_proposal(data)
