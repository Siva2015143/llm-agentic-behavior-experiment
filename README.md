# LLM Agentic Behavior Experiment

**Systematic framework for measuring reasoning behavior in large language models**

This project explores how different prompting strategies affect LLM reasoning patterns, computational cost, and output quality. Built to understand the trade-offs between response speed and reasoning depth.

---

## What It Does

Tests LLMs across three reasoning modes and measures behavioral differences:

| Mode | Approach | Use Case |
|------|----------|----------|
| **P0** | Direct answer, minimal reasoning | Fast responses, simple queries |
| **P3** | Moderate deliberation | Balanced speed and accuracy |
| **P6** | Extended reasoning with self-correction | Complex problems requiring depth |

Tracks token usage, latency, and reasoning patterns to quantify compute-quality trade-offs.

## Quick Start

```bash
git clone https://github.com/Siva2015143/llm-agentic-behavior-experiment
cd web-ui-experiment
pip install -r requirements.txt
```

Add your API key to `.env`:

```
OPENAI_API_KEY=your_key_here
```

Run experiments:

```bash
python src/experiments/agentic_compute_chain_experiment.py
```

Results save to `tmp/agentic_runs/latest/`

## Usage

**Basic Experiment**

```python
from src.utils.llm_provider import get_llm_response

response = get_llm_response(
    model="gpt-3.5-turbo",
    prompt="Solve: What is 25 × 18?",
    thinking_level="P3"
)

print(f"Answer: {response['answer']}")
print(f"Tokens used: {response['tokens']}")
```

**Custom Test Suite**

Edit questions in `agentic_compute_chain_experiment.py`:

```python
TEST_QUESTIONS = [
    "Explain quantum entanglement",
    "Debug this recursive function",
    "Analyze sentiment in customer reviews"
]
```

## Output & Analysis

Generated artifacts:

- `run_logs.jsonl` — Complete experiment data with timestamps, tokens, prompts
- `metrics_summary.csv` — Aggregated performance metrics across modes
- `call_graph.png` — Reasoning pattern visualization (optional)

**Web Dashboard** (optional):

```bash
python webui.py  # http://localhost:7800
```

## What You Learn

**Compute Trade-offs**: Quantify the relationship between reasoning depth and token consumption

**Behavioral Patterns**: Identify how models adjust strategy based on prompting constraints

**Cost Estimation**: Project API costs for different reasoning requirements

**Practical Insights**: Understand when to use fast responses vs deep reasoning

## Technical Details

**Metrics Tracked:**
- Token count per response
- Latency (seconds)
- Reasoning markers ("let me think", "I should verify")
- Cost estimates (based on API pricing)

**Supported Models:**
- OpenAI (GPT-3.5, GPT-4)
- Google Gemini
- Local models via Ollama

## Project Structure

```
├── src/
│   ├── experiments/
│   │   └── agentic_compute_chain_experiment.py
│   └── utils/
│       └── llm_provider.py
├── tmp/agentic_runs/
├── webui.py
└── requirements.txt
```

## Why This Matters

Understanding how LLMs reason under different constraints is critical for:

- **Production Systems**: Optimize cost vs quality for real applications
- **Research**: Study emergent reasoning behaviors systematically
- **Engineering**: Build better prompting strategies based on data

This framework provides reproducible measurements to inform these decisions.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| Missing API key | Create `.env` with valid key |
| No output files | Check `tmp/` exists and is writable |
| Network timeout | Verify API endpoint and connectivity |

## About This Project

Built by an independent researcher exploring LLM behavior through hands-on experimentation. This project is part of my journey learning AI systems, prompt engineering, and experimental design.

**Background**: Fresh graduate researching how LLMs work under the hood. Sharing this to help others studying AI systems and to demonstrate practical ML engineering skills.

**Contact**: Available for collaboration, feedback, and opportunities in ML research or engineering roles.

**Sivamani Battala**  
📧 sivamani6104@gmail.com  
🔗 [GitHub](https://github.com/Siva2015143) | [LinkedIn](https://linkedin.com/in/sivamani-battala)

---

**Stack**: Python, OpenAI API, JSON logging, data visualization  
**Status**: Active development, open to contributions
