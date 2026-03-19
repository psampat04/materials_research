"""LLM prompt construction for proposing and improving descriptor formulas."""

import logging
from dataclasses import dataclass
from pathlib import Path

from llm_client import LLMClient
from prompts import (
    IMPROVEMENT_PROMPT_TEMPLATE,
    INITIAL_PROMPT_TEMPLATE,
    LATEX_PROMPT_TEMPLATE,
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
    missing_required = [k for k in ("function", "explanation") if k not in data]
    if missing_required:
        raise ValueError(
            f"LLM JSON missing required keys: {missing_required}; got: {list(data.keys())}"
        )
    return Proposal(
        function=data["function"],
        explanation=data["explanation"],
        formula="",  # derived post-hoc via derive_latex()
    )


def propose_initial(client: LLMClient) -> Proposal:
    prompt = INITIAL_PROMPT_TEMPLATE.format(problem_desc=PROBLEM_DESCRIPTION)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    data = client.query_json(messages)
    return _build_proposal(data)


def derive_latex(client: LLMClient, code: str) -> str:
    """Ask the LLM to produce an exact LaTeX formula from the Python code."""
    prompt = LATEX_PROMPT_TEMPLATE.format(code=code)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return client.query_text(messages).strip().strip("$")


def propose_improvement(
    client: LLMClient,
    parent_code: str,
    parent_explanation: str,
    metrics_summary: str,
    plot_image: Path,
) -> Proposal:
    prompt = IMPROVEMENT_PROMPT_TEMPLATE.format(
        problem_desc=PROBLEM_DESCRIPTION,
        parent_code=parent_code,
        parent_explanation=parent_explanation,
        metrics_summary=metrics_summary,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    images = []
    if STRUCTURE_FIG.exists():
        images.append(STRUCTURE_FIG)
    if plot_image.exists():
        images.append(plot_image)
    data = client.query_json(messages, images=images if images else None)
    return _build_proposal(data)
