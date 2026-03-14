"""All LLM prompts used in the MCTS formula search agent."""

SYSTEM_PROMPT = "You are a materials science expert. You propose mathematical descriptors and implement them as Python functions."

DEBUG_SYSTEM_PROMPT = "You are a Python debugging expert."

PROBLEM_DESCRIPTION = """
Design a 1-dimensional descriptor for predicting whether ABX3 materials form stable perovskite structures.

The attached figure shows:
(A) The cubic ABX3 perovskite structure — the A cation sits inside a cage of corner-sharing BX6 octahedra. Stability depends on how well the A cation fits this cage, which is controlled by the relative sizes of A, B, and X ions.
(B) A periodic table map of all elements that appear at the A, B, or X sites in the experimentally characterized ABX3 compounds.

Available inputs (all floats):
- rA: Shannon ionic radius of the A-site cation (12-fold coordination)
- rB: Shannon ionic radius of the B-site cation (6-fold coordination)
- rX: Shannon ionic radius of the X-site anion (6-fold coordination)
- nA: oxidation state of A (positive integer)
- nB: oxidation state of B (positive integer)
- nX: oxidation state of X (negative integer)

Physical context:
- The classical Goldschmidt tolerance factor is t = (rA + rX) / (sqrt(2) * (rB + rX))

Requirements:
- The function must be named `descriptor` with signature: def descriptor(rA, rB, rX, nA, nB, nX)
- Return a single finite float value (1-dimensional descriptor)
- The descriptor should be monotonic with respect to stability (either higher = more stable or lower = more stable)

Output format — a JSON object with three keys:
```json
{
  "function": "def descriptor(rA, rB, rX, nA, nB, nX):\\n    ...",
  "explanation": "Brief explanation of the physical reasoning behind this descriptor.",
  "formula": "LaTeX formula, e.g. t = \\\\frac{r_A + r_X}{\\\\sqrt{2}(r_B + r_X)}"
}
```
Output ONLY the JSON, no other text.
""".strip()

INITIAL_PROMPT_TEMPLATE = """
{problem_desc}

Propose a novel descriptor formula. Be creative and try something genuinely new.

STRICT RULES — violating any of these will make your answer invalid:
- Do NOT reproduce the classical Goldschmidt tolerance factor: t = (rA + rX) / (sqrt(2) * (rB + rX))
- Do NOT reproduce the Bartel tau factor: tau = (rX / rB) - nA * (nA - (rA / rB) / ln(rA / rB))
- Your descriptor must use a meaningfully different functional form from both of the above.

Output ONLY the JSON, no other text.
""".strip()

IMPROVEMENT_PROMPT_TEMPLATE = """
{problem_desc}

The previous descriptor was:
- Formula: {parent_formula}
- Explanation: {parent_explanation}
- Function:
```python
{parent_code}
```

Its performance on 576 experimentally characterized ABX3 compounds:
{metrics_summary}

The attached histogram shows the descriptor value distribution for perovskites (blue) vs nonperovskites (orange), with the decision boundary (dashed line). The green region is predicted perovskite, red is predicted nonperovskite.

Analyze what the previous descriptor gets wrong:
- Look at the overlap between perovskite and nonperovskite distributions
- Consider which anion types (O, F, Cl, Br, I) have low accuracy
- Think about what physical effects are missing

Propose an IMPROVED descriptor that addresses these weaknesses.

STRICT RULES — violating any of these will make your answer invalid:
- Do NOT reproduce the classical Goldschmidt tolerance factor: t = (rA + rX) / (sqrt(2) * (rB + rX))
- Do NOT reproduce the Bartel tau factor: tau = (rX / rB) - nA * (nA - (rA / rB) / ln(rA / rB))
- Your new descriptor must have a DIFFERENT functional form from the parent descriptor shown above — different mathematical structure, not just rescaled or rearranged versions of it.
- Do NOT simply rename variables or add a constant offset/scaling to the parent formula.
- Make a genuinely meaningful change: introduce new physical terms, combine variables in a new way, or use a completely different physical rationale.

Output ONLY the JSON, no other text.
""".strip()

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
