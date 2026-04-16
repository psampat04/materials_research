"""Prefix-notation grammar for token-level SR-MCTS.

Formulas are built one token at a time in PREFIX (Polish) notation. We track
how many operand slots are still unfilled so that (a) we know when an
expression is complete and (b) we can prune illegal actions (those that
can't possibly complete inside the remaining budget).
"""

from dataclasses import dataclass

VARS   = ["rA", "rB", "rX", "nA", "nB", "nX"]
UNARY  = ["sqrt", "log", "abs", "square", "neg", "inv"]
BINARY = ["add", "sub", "mul", "div"]
ALL_ACTIONS = VARS + UNARY + BINARY

ARITY = {
    **{v: 0 for v in VARS},
    **{u: 1 for u in UNARY},
    **{b: 2 for b in BINARY},
}


@dataclass(frozen=True)
class State:
    """A partial expression in prefix notation.

    tokens : emitted tokens so far (left-to-right).
    needs  : operand slots still to fill.  `needs == 0` ⇒ terminal.
    """
    tokens: tuple
    needs: int

    @property
    def is_terminal(self) -> bool:
        return self.needs == 0


def initial_state() -> State:
    return State(tokens=(), needs=1)


def apply_action(s: State, action: str) -> State:
    """Consumes one slot and adds ARITY[action] new slots."""
    return State(
        tokens=s.tokens + (action,),
        needs=s.needs - 1 + ARITY[action],
    )


def legal_actions(s: State, max_len: int) -> list[str]:
    """Actions that still allow the expression to complete within max_len.

    Let slack = (max_len - len(tokens)) - needs. After emitting:
      - a variable : slack stays the same
      - a unary    : slack decreases by 1
      - a binary   : slack decreases by 2
    An action is legal iff slack after the action is ≥ 0.
    """
    if s.is_terminal:
        return []
    slack = (max_len - len(s.tokens)) - s.needs
    if slack < 0:
        return []
    actions = list(VARS)
    if slack >= 1:
        actions.extend(UNARY)
    if slack >= 2:
        actions.extend(BINARY)
    return actions


# ---------------------------------------------------------------------------
# Rendering: prefix tokens → Python code / pretty string
# ---------------------------------------------------------------------------

_UNARY_CODE = {
    "sqrt":   "np.sqrt(np.maximum(np.abs({a}), 0.0))",
    "log":    "np.log(np.maximum(np.abs({a}), 1e-9))",
    "abs":    "np.abs({a})",
    "square": "(({a}) ** 2)",
    "neg":    "(-({a}))",
    "inv":    "(1.0 / (np.abs({a}) + 1e-9))",
}

_BINARY_CODE = {
    "add": "(({a}) + ({b}))",
    "sub": "(({a}) - ({b}))",
    "mul": "(({a}) * ({b}))",
    "div": "(({a}) / (np.abs({b}) + 1e-9))",
}

_UNARY_STR = {
    "sqrt":   "√({a})",
    "log":    "log({a})",
    "abs":    "|{a}|",
    "square": "({a})²",
    "neg":    "-({a})",
    "inv":    "1/({a})",
}

_BINARY_STR = {
    "add": "({a} + {b})",
    "sub": "({a} - {b})",
    "mul": "({a} · {b})",
    "div": "({a} / {b})",
}


def _render_prefix(tokens: tuple, unary_fmt: dict, binary_fmt: dict) -> str:
    """Right-to-left stack walk; tolerates partial (mid-search) prefix.

    For complete expressions the stack has exactly the operands each operator
    needs. For partial expressions we substitute '?' for missing operands,
    which makes `tokens_to_pretty` safe to call during search (for logging /
    LLM prompts).
    """
    stack: list[str] = []
    for tok in reversed(tokens):
        if tok in VARS:
            stack.append(tok)
        elif tok in UNARY:
            a = stack.pop() if stack else "?"
            stack.append(unary_fmt[tok].format(a=a))
        elif tok in BINARY:
            a = stack.pop() if stack else "?"
            b = stack.pop() if stack else "?"
            stack.append(binary_fmt[tok].format(a=a, b=b))
    if not stack:
        return ""
    if len(stack) == 1:
        return stack[0]
    return " ; ".join(stack)


def tokens_to_code(tokens: tuple) -> str:
    return _render_prefix(tokens, _UNARY_CODE, _BINARY_CODE)


def tokens_to_pretty(tokens: tuple) -> str:
    return _render_prefix(tokens, _UNARY_STR, _BINARY_STR)


def wrap_as_descriptor(expr_code: str) -> str:
    """Wrap a raw expression in a `descriptor(rA, rB, rX, nA, nB, nX)` function.

    The signature matches `evaluator._exec_descriptor`, so the wrapped code
    can be evaluated by the shared evaluator (plots, CV, per-anion metrics).
    """
    return (
        "def descriptor(rA, rB, rX, nA, nB, nX):\n"
        "    import numpy as np\n"
        f"    return float({expr_code})\n"
    )
