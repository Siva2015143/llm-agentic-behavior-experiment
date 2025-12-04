# 🤖 LLM Agentic Behavior Experiment

**A Simple Tool to Study How AI Models Think and Plan**

> **Measuring how LLMs use different thinking strategies**  
> *Testing models with different reasoning constraints to see how they behave.*

---

## 📋 What This Project Does

This project tests AI models with three types of instructions:
- **P0**: Give a direct answer (no extra thinking)
- **P3**: Think a bit before answering (some reasoning)
- **P6**: Think as much as needed (full reasoning, tools, self-correction)

For each test, we measure:
- How many words (tokens) the model uses
- How long it takes
- What thinking patterns it shows
- Estimated computation cost

---

## 📁 Project Structure
web-ui-experiment/
├── README.md
├── requirements.txt
├── webui.py
│
├── src/
│ ├── experiments/agentic_compute_chain_experiment.py
│ └── utils/ (config, llm_provider, metrics_wrapper, utils)
│
├── analysis/sample_logs/ (example results)
├── results/ (graphs and summaries)
└── exports/ (ready-to-share data)

---

## 🚀 Quick Start

### 1. Setup
git clone https://github.com/yourusername/web-ui-experiment.git
cd web-ui-experiment
pip install -r requirements.txt

2. Add API Key
Create .env file:
OPENAI_API_KEY=your_key_here

3. Run Tests
python src/experiments/agentic_compute_chain_experiment.py
Results save to: tmp/agentic_runs/latest/

📊 Key Findings
Thinking Costs More
P0: Least computation (fastest)
P6: 3-4× more computation than P0 (but better answers)
Trade-off: Better answers need more thinking time

Common Thinking Patterns
Pattern	                 How Often (P6)	             Example
Step-by-step thinking	    87%	               "Let me think step by step"
Using tools	                62%	               "I should search for..."
Self-correction         	45%	               "Actually, let me correct..."
Asking for help	            38%	               "I'll ask a helper..."
Thinking Has Limits
P0: No rethinking

P3: 1-2 rethinks on average

P6: Usually 3-4 rethinks (max 7)


🔧 How It Works
1. Testing Method
Same question asked three ways:
P0: "Just answer"
P3: "Think a bit then answer"
P6: "Think as much as you need"

2. What We Track
Words used (tokens)
Time taken
Special phrases ("let me think", "I should search")
Thinking patterns

3. Results We Create
Comparison tables
Cost vs thinking graphs
Thinking pattern diagrams


💡 Why This Matters
For Learning
Shows how to:
-Measure AI behavior systematically
-Understand AI system trade-offs
-Collect and analyze experiment data

For Practical Use
Helps you:
-Choose the right thinking level for your task
-Estimate AI system costs
-Debug unexpected AI answers


🧪 Run Your Own Tests
Simple Example
from src.utils.llm_provider import get_llm_response
response = get_llm_response(
    model="gpt-3.5-turbo",
    prompt="What is 25 × 18?",
    thinking_level="P3"
)
print(f"Answer: {response['answer']}")
print(f"Tokens: {response['tokens']}")
Add Your Questions
Edit in 'agentic_compute_chain_experiment.py':


TEST_QUESTIONS = [
    "Explain photosynthesis simply",
    "Plan a Japan trip itinerary",
    "Debug this Python code"
]


📈 View Results
Console Output
  Running P0 test... ✓ (1.2s, 45 tokens)
  Running P3 test... ✓ (3.8s, 120 tokens)
  Running P6 test... ✓ (8.5s, 310 tokens)


Generated Files
-run_logs.jsonl - All details
-call_graph.png - Thinking pattern picture
-metrics_summary.csv - Comparison numbers


Web Interface (Optional)
python webui.py
Open: http://localhost:7800


❓ FAQ
Q: Do I need paid API keys?
A: No! You can use free tiers: OpenAI ($5 free), Gemini (free), or local Ollama models.

Q: I'm new to Python. Can I use this?
A: Yes! If you can run pip install and python file.py, you're ready.

Q: Can I add my own questions?
A: Yes! Just edit the question list and run the experiment.

Q: Why P0, P3, P6 names?
A: Just labels: P0=no thinking, P6=max thinking, P3=in between.

🛠️ Troubleshooting
Problem	                       Fix
"Module not found"   	Run pip install -r requirements.txt
"API key missing"	    Create .env file with your key
Script stops early   	Check internet connection
No results saved	    Check if tmp/ folder exists


👨💻 About
I'm an independent researcher exploring how AI systems work. This project is my hands-on learning about:
-AI behavior patterns
-Experiment methods
-Data analysis
-Clear documentation
This is a learning project - sharing to help others learn too.


📄 License
MIT License - free to use, modify, and share.


🙏 Thanks
Thanks to the open source community and everyone who shares knowledge freely.