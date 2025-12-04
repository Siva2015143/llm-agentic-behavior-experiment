
# ===========================================================
# agentic_compute_chain_experiment.py - ENHANCED VERSION
# ===========================================================
# PURPOSE:
#   Advanced behavior-driven experiment to measure Agentic Compute Dynamics
#   Tracking reasoning steps, tool calls, recursive decisions, and compute propagation
#
# DESIGN:
#   - Model decides when to stop (no fixed steps)
#   - Advanced agentic behavior detection (spawns, tools, self-recursion)
#   - Comprehensive compute dynamics measurement
#   - Real-time behavior classification and metrics
# ===========================================================

import os
import json
import time
import uuid
import random
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage
from src.utils.llm_provider import get_llm_model, estimate_flops

# ==========================================================
# 🔬 Enhanced Agentic Compute Dynamics Experiment Runner
# ==========================================================

RUN_ID = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join("tmp", "agentic_runs", RUN_ID)
os.makedirs(LOG_DIR, exist_ok=True)

# 🧩 Experiment setup
MODELS = [("google", "gemini-2.0-flash")]

USER_TASK = (
    "Research and summarize the evolution of AI scaling laws, "
    "then propose 2 new experimental metrics for compute-efficiency forecasting."
)

PROMPT_VARIANTS = [
    {
        "id": "P0",
        "desc": "Strict, minimal reasoning — single pass only.",
        "system": "You are a single-step assistant. Never delegate or call other models."
    },
    {
        "id": "P3", 
        "desc": "Moderate — allowed to re-evaluate results and ask itself one helper query if needed.",
        "system": "You can re-evaluate your output and perform limited self-assistance."
    },
    {
        "id": "P6",
        "desc": "Fully autonomous — freely spawn helper models to refine reasoning before final answer.",
        "system": "You can autonomously generate helper queries to improve your final output."
    },
]

# Control knobs
TRIALS_PER_VARIANT = 3
MAX_CALLS = 50
DELAY_BETWEEN_CALLS = 1.5

# 🎯 Agentic Behavior Patterns
AGENTIC_PATTERNS = {
    "helper_spawn": [
        "helper", "assistant", "subagent", "delegate", "spawn", "call another",
        "ask an expert", "consult", "get help from"
    ],
    "self_recursion": [
        "let me think", "step back", "re-evaluate", "reconsider", "think again",
        "let me reconsider", "on second thought", "actually", "correction"
    ],
    "planning": [
        "plan", "strategy", "approach", "break down", "steps", "first", "next",
        "then", "finally", "roadmap", "outline"
    ],
    "tool_usage": [
        "search", "calculate", "compute", "look up", "find", "retrieve",
        "fetch", "query", "tool", "function"
    ],
    "completion": [
        "final answer", "conclusion", "summary", "done", "complete", "finished",
        "in summary", "to conclude", "overall"
    ]
}

class AgenticBehaviorAnalyzer:
    """Advanced analyzer for detecting agentic compute dynamics"""
    
    def __init__(self):
        self.reasoning_chain = []
        self.behavior_history = []
        
    def detect_behavior(self, text: str, response_obj: Any) -> Dict[str, Any]:
        """Comprehensive agentic behavior detection"""
        text_lower = text.lower()
        
        behavior = {
            "spawn_type": "direct_response",  # default
            "tool_calls": self._extract_tool_calls(response_obj),
            "reasoning_depth": len(self.reasoning_chain),
            "is_self_spawn": False,
            "confidence": 0.0,
            "trigger_phrases": []
        }
        
        # Check each behavior pattern
        detected_patterns = []
        
        # Tool calls (highest priority)
        if behavior["tool_calls"]:
            behavior["spawn_type"] = "tool_invocation"
            detected_patterns.append("tool_usage")
            behavior["confidence"] = 0.9
            
        # Helper spawning
        elif self._pattern_match(text_lower, AGENTIC_PATTERNS["helper_spawn"]):
            behavior["spawn_type"] = "helper_spawn"
            detected_patterns.append("helper_spawn")
            behavior["confidence"] = 0.8
            
        # Self-recursion
        elif self._pattern_match(text_lower, AGENTIC_PATTERNS["self_recursion"]):
            behavior["spawn_type"] = "self_recursion" 
            behavior["is_self_spawn"] = True
            detected_patterns.append("self_recursion")
            behavior["confidence"] = 0.7
            self.reasoning_chain.append("self_recursion")
            
        # Planning behavior
        elif self._pattern_match(text_lower, AGENTIC_PATTERNS["planning"]):
            behavior["spawn_type"] = "planning"
            detected_patterns.append("planning")
            behavior["confidence"] = 0.6
            
        # Extract trigger phrases
        behavior["trigger_phrases"] = self._extract_trigger_phrases(text_lower)
        
        # Update reasoning chain
        self._update_reasoning_chain(behavior, detected_patterns)
        
        self.behavior_history.append(behavior)
        return behavior
    
    def _pattern_match(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any patterns"""
        return any(pattern in text for pattern in patterns)
    
    def _extract_tool_calls(self, response_obj: Any) -> List[str]:
        """Extract tool/function calls from response"""
        tool_calls = []
        
        # Check for tool_calls attribute
        if hasattr(response_obj, 'tool_calls') and response_obj.tool_calls:
            tool_calls.extend([str(tc) for tc in response_obj.tool_calls])
            
        # Check for function calls in text
        if hasattr(response_obj, 'content'):
            content = response_obj.content
            # Look for function call patterns
            function_patterns = [
                r"search\([^)]*\)", r"calculate\([^)]*\)", r"lookup\([^)]*\)",
                r"tool_[a-z_]*\([^)]*\)", r"function_[a-z_]*\([^)]*\)"
            ]
            for pattern in function_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                tool_calls.extend(matches)
                
        return tool_calls
    
    def _extract_trigger_phrases(self, text: str) -> List[str]:
        """Extract phrases that triggered behavior detection"""
        triggers = []
        for behavior_type, patterns in AGENTIC_PATTERNS.items():
            for pattern in patterns:
                if pattern in text:
                    triggers.append(f"{behavior_type}:{pattern}")
        return triggers
    
    def _update_reasoning_chain(self, behavior: Dict, patterns: List[str]):
        """Update reasoning depth based on current behavior"""
        if behavior["spawn_type"] in ["helper_spawn", "tool_invocation"]:
            # Reset chain for external calls
            self.reasoning_chain = ["external_spawn"]
        elif behavior["spawn_type"] == "self_recursion":
            # Continue self-recursion chain
            if not self.reasoning_chain or self.reasoning_chain[-1] != "self_recursion":
                self.reasoning_chain.append("self_recursion")
        elif behavior["spawn_type"] == "direct_response":
            # Break chain if no agentic behavior
            if len(self.reasoning_chain) > 3:
                self.reasoning_chain = []

def should_terminate_trial(behavior_analyzer: AgenticBehaviorAnalyzer, 
                          step: int, 
                          total_calls: int,
                          recent_text: str) -> bool:
    """Intelligent termination logic based on agentic dynamics"""
    
    # Safety: Maximum calls reached
    if step >= MAX_CALLS - 1:
        return True
        
    # Completion detection
    text_lower = recent_text.lower()
    if any(completion in text_lower for completion in AGENTIC_PATTERNS["completion"]):
        return True
        
    # Stagnation: No agentic behavior in last 5 calls
    recent_behaviors = behavior_analyzer.behavior_history[-5:] if len(behavior_analyzer.behavior_history) >= 5 else behavior_analyzer.behavior_history
    if recent_behaviors and not any(b["spawn_type"] != "direct_response" for b in recent_behaviors):
        if step > 10:  # Only apply after sufficient exploration
            return True
            
    # Excessive self-recursion (safety cutoff)
    self_recursion_count = sum(1 for b in behavior_analyzer.behavior_history if b["is_self_spawn"])
    if self_recursion_count > 8:
        return True
        
    # Original probabilistic termination (reduced probability)
    if random.random() < 0.15:  # Reduced from 0.25 to allow more agentic behavior
        return True
        
    return False

def run_trial(provider: str, model_name: str, variant: Dict) -> Dict[str, Any]:
    """Enhanced trial runner with comprehensive agentic dynamics tracking"""
    
    trial_id = str(uuid.uuid4())
    trial_log_path = os.path.join(LOG_DIR, f"{variant['id']}_{trial_id}.jsonl")
    
    model = get_llm_model(provider=provider, model_name=model_name)
    behavior_analyzer = AgenticBehaviorAnalyzer()
    
    call_graph = {
        "nodes": [],
        "edges": [],
        "metadata": {
            "trial_id": trial_id,
            "variant": variant["id"],
            "model": model_name,
            "start_time": datetime.utcnow().isoformat()
        }
    }
    
    parent_stack = [None]
    total_flops = 0
    total_tokens = 0
    agentic_metrics = {
        "total_spawns": 0,
        "tool_calls": 0,
        "self_recursions": 0,
        "planning_actions": 0,
        "max_reasoning_depth": 0
    }

    print(f"\n🚀 Running {variant['id']} | Model: {model_name}")

    for step in range(MAX_CALLS):
        prompt = f"{variant['system']}\n\nUser Task:\n{USER_TASK}\n\nCall #{step + 1}"
        start_time = time.time()

        # Invoke model
        try:
            response = model.invoke([HumanMessage(content=prompt)])
        except Exception as e:
            print(f"[ERROR] Model invoke failed: {e}")
            break

        latency = time.time() - start_time

        # Extract response text
        if hasattr(response, "content"):
            text = response.content
        elif isinstance(response, str):
            text = response
        else:
            text = str(response)

        tokens_in = len(prompt.split())
        tokens_out = len(text.split())
        flops = estimate_flops(tokens_in + tokens_out, model_name)
        total_flops += flops
        total_tokens += tokens_in + tokens_out

        # Advanced behavior analysis
        behavior = behavior_analyzer.detect_behavior(text, response)
        
        # Update agentic metrics
        if behavior["spawn_type"] == "helper_spawn":
            agentic_metrics["total_spawns"] += 1
        elif behavior["spawn_type"] == "tool_invocation":
            agentic_metrics["tool_calls"] += len(behavior["tool_calls"])
        elif behavior["spawn_type"] == "self_recursion":
            agentic_metrics["self_recursions"] += 1
        elif behavior["spawn_type"] == "planning":
            agentic_metrics["planning_actions"] += 1
            
        agentic_metrics["max_reasoning_depth"] = max(
            agentic_metrics["max_reasoning_depth"], 
            behavior["reasoning_depth"]
        )

        # Build comprehensive call data
        call_data = {
            "call_id": step + 1,
            "parent_id": parent_stack[-1],
            "behavior_type": behavior["spawn_type"],
            "tool_calls": behavior["tool_calls"],
            "reasoning_depth": behavior["reasoning_depth"],
            "is_self_spawn": behavior["is_self_spawn"],
            "confidence": behavior["confidence"],
            "trigger_phrases": behavior["trigger_phrases"],
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency": latency,
            "GFLOPs": flops,
            "timestamp": datetime.utcnow().isoformat(),
            "content_preview": text[:120]
        }

        call_graph["nodes"].append(call_data)

        # Update call hierarchy based on spawn behavior
        if behavior["spawn_type"] in ["helper_spawn", "tool_invocation"]:
            # Create edge for spawn relationship
            call_graph["edges"].append({
                "from": parent_stack[-1],
                "to": step + 1,
                "type": behavior["spawn_type"],
                "timestamp": datetime.utcnow().isoformat()
            })
            parent_stack.append(step + 1)

        # Check for termination
        if should_terminate_trial(behavior_analyzer, step, len(call_graph["nodes"]), text):
            break

        time.sleep(DELAY_BETWEEN_CALLS)

    # Calculate derived metrics
    total_calls = len(call_graph["nodes"])
    compute_amplification = total_flops / (tokens_in * estimate_flops(1, model_name)) if total_calls > 0 else 0
    
    agentic_metrics.update({
        "total_calls": total_calls,
        "agentic_call_ratio": (agentic_metrics["total_spawns"] + agentic_metrics["tool_calls"]) / total_calls if total_calls > 0 else 0,
        "compute_amplification": compute_amplification,
        "average_reasoning_depth": sum(node["reasoning_depth"] for node in call_graph["nodes"]) / total_calls if total_calls > 0 else 0
    })

    # Write enhanced logs
    with open(trial_log_path, "w", encoding="utf-8") as f:
        for node in call_graph["nodes"]:
            f.write(json.dumps(node) + "\n")

    # Write call graph for analysis
    graph_path = os.path.join(LOG_DIR, f"{variant['id']}_{trial_id}_graph.json")
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(call_graph, f, indent=2)

    print(f"✅ {variant['id']} complete: {total_calls} calls, "
          f"{round(total_flops, 2)} GFLOPs, "
          f"{agentic_metrics['total_spawns']} spawns, "
          f"{agentic_metrics['tool_calls']} tools")

    return {
        "trial_id": trial_id,
        "variant": variant["id"],
        "model": model_name,
        "total_calls": total_calls,
        "total_flops": total_flops,
        "total_tokens": total_tokens,
        "agentic_metrics": agentic_metrics,
        "call_graph_path": graph_path
    }

def main():
    """Enhanced main function with comprehensive analytics"""
    
    all_results = []
    agentic_summary = {
        "experiment_id": RUN_ID,
        "start_time": datetime.utcnow().isoformat(),
        "variants": {}
    }

    for provider, model_name in MODELS:
        for variant in PROMPT_VARIANTS:
            variant_results = []
            print(f"\n🔬 Testing Variant {variant['id']}: {variant['desc']}")
            
            for t in range(TRIALS_PER_VARIANT):
                print(f"  Trial {t+1}/{TRIALS_PER_VARIANT}...")
                result = run_trial(provider, model_name, variant)
                variant_results.append(result)
                all_results.append(result)
                
            # Calculate variant-level metrics
            if variant_results:
                agentic_summary["variants"][variant["id"]] = {
                    "trials": len(variant_results),
                    "avg_calls": sum(r["total_calls"] for r in variant_results) / len(variant_results),
                    "avg_flops": sum(r["total_flops"] for r in variant_results) / len(variant_results),
                    "avg_spawns": sum(r["agentic_metrics"]["total_spawns"] for r in variant_results) / len(variant_results),
                    "avg_tools": sum(r["agentic_metrics"]["tool_calls"] for r in variant_results) / len(variant_results),
                    "avg_amplification": sum(r["agentic_metrics"]["compute_amplification"] for r in variant_results) / len(variant_results)
                }

    # Write comprehensive results
    out_path = os.path.join(LOG_DIR, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "raw_results": all_results,
            "agentic_summary": agentic_summary
        }, f, indent=2)

    # Print executive summary
    print(f"\n📊 EXPERIMENT COMPLETE")
    print(f"📁 Logs saved to: {LOG_DIR}")
    print(f"\n🎯 AGENTIC BEHAVIOR SUMMARY:")
    for variant_id, stats in agentic_summary["variants"].items():
        print(f"   {variant_id}: {stats['avg_calls']:.1f} calls, "
              f"{stats['avg_spawns']:.1f} spawns, "
              f"{stats['avg_tools']:.1f} tools, "
              f"{stats['avg_amplification']:.1f}x compute")

if __name__ == "__main__":
    main()