# LLM Agentic Behavior Experiment

**A simple tool to study how AI models think and plan**

Measuring how LLMs use different thinking strategies by testing models with various reasoning constraints to observe behavioral patterns.

---

## What This Project Does

This project tests AI models using three distinct instruction types:

- **P0** — Direct answers without extra reasoning
- **P3** — Moderate thinking before responding
- **P6** — Unrestricted reasoning with tools and self-correction

Each test measures token usage, response time, thinking patterns, and estimated computational cost.

---

## Quick Start

**Setup**

```bash
git clone https://github.com/Siva2015143/llm-agentic-behavior-experiment
cd web-ui-experiment
pip install -r requirements.txt
```

**Add API Key**

Create a `.env` file:

```
OPENAI_API_KEY=your_key_here
```

**Run Tests**

```bash
python src/experiments/agentic_compute_chain_experiment.py
```

Results are saved to: `tmp/agentic_runs/latest/`

---

## Key Findings

**Thinking Costs More**

- P0: Minimal computation (fastest)
- P6: 3-4× more computation than P0 (higher quality answers)
- Trade-off: Better responses require more thinking time

**Common Thinking Patterns**

|          Pattern      |   Frequency (P6)   |          Example              |
|-----------------------|--------------------|-------------------------------|
| Step-by-step thinking |         87%        | "Let me think step by step"   |
| Tool usage            |         62%        | "I should search for..."      |
| Self-correction       |         45%        | "Actually, let me correct..." |
| Delegation            |         38%        | "I'll ask a helper..."        |

**Thinking Has Limits**

- P0: No rethinking capability
- P3: 1-2 iterations on average
- P6: Typically 3-4 iterations (max 7 observed)

---

## How It Works

**Testing Method**

The same question is asked three different ways:

- P0: "Just answer"
- P3: "Think a bit then answer"
- P6: "Think as much as you need"

**Metrics Tracked**

- Token count (words used)
- Response time
- Special reasoning phrases
- Behavioral patterns

**Output Generated**

- Comparison tables
- Cost vs thinking graphs
- Thinking pattern visualizations

---

## Why This Matters

**For Learning**

Shows how to measure AI behavior systematically, understand system trade-offs, and collect experimental data properly.

**For Practical Use**

Helps you choose appropriate thinking levels for tasks, estimate system costs, and debug unexpected responses.

---

## Run Your Own Tests

**Simple Example**

```python
from src.utils.llm_provider import get_llm_response

response = get_llm_response(
    model="gpt-3.5-turbo",
    prompt="What is 25 × 18?",
    thinking_level="P3"
)

print(f"Answer: {response['answer']}")
print(f"Tokens: {response['tokens']}")
```

**Add Your Questions**

Edit in `agentic_compute_chain_experiment.py`:

```python
TEST_QUESTIONS = [
    "Explain photosynthesis simply",
    "Plan a Japan trip itinerary",
    "Debug this Python code"
]
```

---

## View Results

**Console Output**

```
Running P0 test... ✓ (1.2s, 45 tokens)
Running P3 test... ✓ (3.8s, 120 tokens)
Running P6 test... ✓ (8.5s, 310 tokens)
```

**Generated Files**

- `run_logs.jsonl` — Detailed experiment logs
- `call_graph.png` — Visual thinking patterns
- `metrics_summary.csv` — Numerical comparisons

**Web Interface (Optional)**

```bash
python webui.py
```

Open: http://localhost:7800

---

## FAQ

**Do I need paid API keys?**  
No. You can use free tiers from OpenAI ($5 credit), Gemini, or local Ollama models.

**I'm new to Python. Can I use this?**  
Yes. If you can run `pip install` and `python file.py`, you're ready.

**Can I add my own questions?**  
Yes. Just edit the question list and rerun the experiment.

**Why the P0, P3, P6 naming?**  
Simple labels where P0 = no thinking, P6 = maximum thinking, P3 = balanced.

---

## Troubleshooting

|      Problem            |             Solution                  |
|-------------------------|---------------------------------------|
| "Module not found"      | Run `pip install -r requirements.txt` |
| "API key missing"       | Create `.env` file with your key      |
| Script stops early      | Check internet connection             |
| No results saved        | Verify `tmp/` folder exists           |

---

## About

I'm an independent researcher and fresher exploring AI system behavior through hands-on experimentation. This project examines AI reasoning patterns, experimental methodologies, and data analysis techniques.

Built this as a learning project while studying how LLMs work. Sharing it to help others who are also learning about AI systems.

---

## Acknowledgments

Thanks to the open source community for sharing knowledge and enabling collaborative learning.
