# Research: Material Science

**Goal**: build a general-purpose LLM-guided MCTS agent for automatic discovery of mathematical descriptors (symbolic descriptor discovery / Collective Variables) in materials science. The workflow is:

1. Give the LLM a problem description
2. The LLM proposes candidate formulas (Python function + LaTeX + explanation + etc.)
3. Evaluate each formula's classification ability on labeled data using decision trees, SVMs, etc.
4. Feed evaluation results (accuracy, histograms, parity plots) back to the LLM to guide improvement
5. Use MCTS to systematically explore and optimize the formula space

The aim is to leverage the prior knowledge already embedded in sota LLMs.

**Immediate target**: reproduce and surpass the perovskite stability tolerance factor τ proposed by Bartel et al. (2019). τ achieves about 92% classification accuracy on 576 ABX₃ compounds, outperforming the classical Goldschmidt tolerance factor t (about 74%). See `bartel2019-new-tolerance-factor.md` for details.

## Setup

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/rbamzai21/research_material_science.git

# If already cloned without submodules
git submodule update --init --recursive

# Python environment (requires uv: https://docs.astral.sh/uv/)
uv sync --all-extras

# Pre-commit hooks
uv run pre-commit install
```

## Formula search agent

MCTS-guided LLM agent that proposes, evaluates, and iteratively improves descriptor formulas.

```bash
# Set your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env

# Run the search (default: 50 formula budget, GPT-4.1 Nano)
uv run python run_search.py

# Resume an interrupted search
uv run python run_search.py search.resume=search_runs/<run_name>/search_state.json

# Override config
uv run python run_search.py mcts.budget=100 llm.model=gpt-4o-mini
```

Results are saved under `search_runs/<timestamp>/` with plots and a JSON state file.

## Project structure

```
├── perovskite-stability/      # Submodule: Bartel et al. data & code
├── reproduce_evidence/         # Scripts reproducing paper figures
│   ├── evidence1_abx3_classification.py   # Fig 2 A, B, C
│   ├── evidence2_double_perovskites.py    # ICSD double perovskites
│   └── evidence3_dft_correlation.py       # Fig 2 D
├── run_search.py               # MCTS formula search entry point
├── mcts.py                     # MCTS algorithm (UCB1, expand, backprop)
├── evaluator.py                # Evaluate formulas on 576 ABX3
├── proposer.py                 # LLM prompt construction (text + vision)
├── debugger.py                 # Error recovery loop
├── llm_client.py               # OpenAI API wrapper (text + vision)
├── state.py                    # Persistent search state
├── conf/config.yaml            # Hydra configuration
├── legacy/                     # Previous prototype code
├── bartel2019-new-tolerance-factor.md
├── pyproject.toml
└── .pre-commit-config.yaml
```
