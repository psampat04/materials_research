"""Error recovery loop for generated descriptor functions."""

import logging
import re

from llm_client import LLMClient

DEBUG_SYSTEM_PROMPT = "You are a Python debugging expert."

DEBUG_PROMPT_TEMPLATE = """
The following Python descriptor function crashed when executed on the dataset:

```python
{code}
```

Error message:
```
{error}
```

Fix the function so it handles all valid inputs without errors.
Constraints:
- The function must be named `descriptor`
- Signature: def descriptor(rA, rB, rX, nA, nB, nX)
- rA, rB, rX are positive floats (ionic radii in Angstroms)
- nA, nB are positive integers (oxidation states)
- nX is a negative integer (oxidation state)
- rA > rB is guaranteed
- Must return a single finite float

Output ONLY the fixed Python function, no explanation.
""".strip()

log = logging.getLogger(__name__)


def _extract_function_raw(response: str) -> str:
    """Extract a Python function from a plain-text LLM response (no JSON)."""
    match = re.search(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
    code = match.group(1).strip() if match else response.strip()

    if "def descriptor" not in code:
        raise ValueError("No 'def descriptor' found in LLM response")

    lines = code.split("\n")
    func_lines = []
    in_func = False
    for line in lines:
        if line.strip().startswith("def descriptor"):
            in_func = True
        if in_func:
            if (
                func_lines
                and line.strip()
                and not line[0].isspace()
                and not line.strip().startswith("def")
            ):
                break
            func_lines.append(line)

    if not func_lines:
        raise ValueError("Could not extract function body")

    return "\n".join(func_lines)


def debug_function(
    client: LLMClient,
    code: str,
    error: str,
) -> str:
    """Try to fix a crashing function by sending the error back to the LLM."""
    prompt = DEBUG_PROMPT_TEMPLATE.format(code=code, error=error)
    messages = [
        {"role": "system", "content": DEBUG_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    response = client.query_text(messages)
    return _extract_function_raw(response)
