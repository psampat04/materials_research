"""Prompts for LLM-SR style search with skeleton + params."""

SYSTEM_PROMPT = "You are a materials science expert. You propose mathematical descriptor skeletons implemented as Python functions."

PROBLEM_DESCRIPTION = """
Design a 1-dimensional descriptor skeleton for predicting whether ABX3 materials form stable perovskite structures.

Available inputs:
- rA: Shannon ionic radius of the A-site cation (12-fold coordination)
- rB: Shannon ionic radius of the B-site cation (6-fold coordination)
- rX: Shannon ionic radius of the X-site anion (6-fold coordination)
- nA: oxidation state of A (positive integer)
- nB: oxidation state of B (positive integer)
- nX: oxidation state of X (negative integer)
- params: list of exactly 1 or 2 numeric scaling constants — use params[0] and optionally params[1] only

Physical context:
- The dataset spans five anion families: oxides (O²⁻), fluorides (F⁻), chlorides (Cl⁻), bromides (Br⁻), iodides (I⁻). A good descriptor must work uniformly across all five.
- The classical Goldschmidt tolerance factor is t = (rA + rX) / (sqrt(2) * (rB + rX))
- Your skeleton's STRUCTURE is what matters — the numeric values in params[] will be fitted automatically by an optimizer. Do NOT hardcode numbers like 1.5 or 0.7; use params[0], params[1], etc. instead.

Requirements:
- Function signature: def descriptor(rA, rB, rX, nA, nB, nX, params)
- Use params[0], params[1], ... as scaling constants
- Return a single finite float
- The descriptor should be monotonic with respect to stability
- Keep it simple: 3-6 operations max. A good skeleton fits on a napkin.

Output format — a JSON object with three keys:
```json
{
  "function": "def descriptor(rA, rB, rX, nA, nB, nX, params):\\n    ...",
  "explanation": "Brief explanation of the physical reasoning.",
  "formula": "LaTeX formula using r_A, r_B, r_X, n_A, n_B, n_X, p_0, p_1, ..."
}
```
Output ONLY the JSON, no other text.
""".strip()

INITIAL_SKELETON_TEMPLATE = """
{problem_desc}

Propose a novel descriptor skeleton. Be creative — try something genuinely new.

STRICT RULES — violating any of these will make your answer invalid:
- Do NOT reproduce t or tau exactly as standalone formulas (you MAY incorporate them as components inside a larger expression)
- Your descriptor must add physical meaning beyond simple size ratios — use oxidation states, charge-radius coupling, or other effects
- Do NOT hardcode numeric constants — use params[0], params[1], etc. for all scaling values

Output ONLY the JSON, no other text.
""".strip()

IMPROVEMENT_SKELETON_TEMPLATE = """
{problem_desc}

Here are the {k} best-performing descriptor skeletons found so far:

{examples}

Propose a NEW skeleton inspired by these but with a DIFFERENT functional form — not just a rescaling or rearrangement of one of the examples above.

Think about:
- What physical effects might these skeletons be missing?
- Which anion types (O, F, Cl, Br, I) are likely hardest to classify and why?
- Can you combine elements from different examples in a novel way?

STRICT RULES — violating any of these will make your answer invalid:
- Do NOT reproduce t or tau exactly as standalone formulas (you MAY incorporate them as components)
- Your new skeleton must have a genuinely different mathematical structure
- Do NOT hardcode numeric constants — use params[0], params[1], etc.

Output ONLY the JSON, no other text.
""".strip()