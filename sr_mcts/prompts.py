"""Prompt template for the SR-MCTS LLM-policy prior."""

SYSTEM_PROMPT = (
    "You are a materials-science expert guiding a symbolic search for a "
    "descriptor that classifies ABX3 perovskite stability. You rate candidate "
    "next tokens for a partial formula based on physical plausibility and "
    "likelihood of completing into a useful expression."
)

RATE_TOKEN_TEMPLATE = """You are building a formula in PREFIX (Polish) notation.

Variables:
  rA, rB, rX  — ionic radii (Å) of A, B, X sites  (rA > rB typically)
  nA, nB, nX  — formal charges of A, B, X         (nA, nB > 0; nX < 0)

Operators:
  unary  (arity 1): sqrt, log, abs, square, neg, inv
  binary (arity 2): add, sub, mul, div

Context: Bartel's tau factor  rX/rB - nA·(nA - (rA/rB)/ln(rA/rB))  reaches ~92%
accuracy, so ratios of radii (rX/rB, rA/rB) and interactions between radii and
charges are physically promising.

Current partial expression : {partial}
Reads as                   : {pretty}
Operand slots still needed : {needs}

Candidate next tokens: [{action_list}]

For EACH candidate above, give an integer score from 1 (bad choice here) to 10
(excellent choice here). Consider physical intuition AND what naturally
completes the expression given the slots remaining.

Respond with ONLY valid JSON in exactly this shape:
{{"ratings": {{"<token>": <int 1-10>, ...}}}}
Include every candidate token exactly once.
"""
