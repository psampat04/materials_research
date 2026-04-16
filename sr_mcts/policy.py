"""Policies for SR-MCTS.

A *policy* maps a (State, legal_actions) pair to a dict of action→probability.
The MCTS loop uses two policies:
  * `expand_policy`   — seeds priors for a new tree node (can be expensive).
  * `rollout_policy`  — completes partial expressions cheaply.

Using the LLM only at expansion bounds total API cost by the tree size
(≈ one call per unique partial expression we ever expand), independent of
the number of MCTS iterations.
"""

import json
import logging

import numpy as np

from llm_client import LLMClient
from sr_mcts.grammar import State, tokens_to_pretty
from sr_mcts.prompts import RATE_TOKEN_TEMPLATE, SYSTEM_PROMPT

log = logging.getLogger(__name__)


def uniform_policy(state: State, actions: list[str]) -> dict[str, float]:
    p = 1.0 / len(actions)
    return {a: p for a in actions}


class LLMPolicy:
    """LLM-guided prior: asks the model to rate each candidate next token.

    Ratings are softmaxed into probabilities. Results are cached per
    (partial_tokens, action_set) so the same state never triggers two LLM
    calls. Any failure (timeout, bad JSON, missing keys) gracefully falls
    back to the uniform prior so search always makes progress.
    """

    def __init__(
        self,
        client: LLMClient,
        softmax_temperature: float = 2.0,
    ) -> None:
        self.client = client
        self.softmax_temperature = softmax_temperature
        self.cache: dict[tuple, dict[str, float]] = {}
        self.calls = 0
        self.failures = 0

    def __call__(self, state: State, actions: list[str]) -> dict[str, float]:
        key = (state.tokens, tuple(sorted(actions)))
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        prompt = self._build_prompt(state, actions)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        try:
            self.calls += 1
            data = self.client.query_json(messages)
            ratings = data.get("ratings", {})
            scores = np.array(
                [float(ratings.get(a, 5.0)) for a in actions], dtype=float
            )
            z = scores / self.softmax_temperature
            z = z - z.max()
            p = np.exp(z)
            p = p / p.sum()
            result = {a: float(p[i]) for i, a in enumerate(actions)}
        except Exception as e:
            self.failures += 1
            log.warning(
                "LLM prior failed at '%s' (%s); using uniform fallback.",
                " ".join(state.tokens) or "(start)", e,
            )
            result = uniform_policy(state, actions)

        self.cache[key] = result
        return result

    @staticmethod
    def _build_prompt(state: State, actions: list[str]) -> str:
        partial = " ".join(state.tokens) if state.tokens else "(empty — start a new formula)"
        pretty  = tokens_to_pretty(state.tokens) if state.tokens else "—"
        return RATE_TOKEN_TEMPLATE.format(
            partial=partial,
            pretty=pretty,
            needs=state.needs,
            action_list=", ".join(actions),
        )
