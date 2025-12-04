import time
import csv
import os
from datetime import datetime

def log_metrics(model_name, input_toks, output_toks, latency):
    total_toks = input_toks + output_toks
    params = 1.8e11  # ~180B for Gemini-2.0-Flash
    flops = 6 * params * total_toks

    log_path = os.path.join(os.getcwd(), "agent_metrics.csv")
    new_file = not os.path.exists(log_path)

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "model", "input_toks", "output_toks",
                             "total_toks", "latency_s", "flops"])
        writer.writerow([
            datetime.utcnow().isoformat(), model_name, input_toks, output_toks,
            total_toks, round(latency, 3), f"{flops:.3e}"
        ])

    print(f"""
==========================
📊 Agentic Call Metrics
Model: {model_name}
Input Tokens: {input_toks}
Output Tokens: {output_toks}
Latency: {latency:.2f}s
Total Tokens: {total_toks}
Estimated FLOPs: {flops:.3e}
==========================
""")
