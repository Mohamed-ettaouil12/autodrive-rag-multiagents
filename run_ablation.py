#!/usr/bin/env python3
"""
Ablation Study Runner - Executes golden set with each flag combination
and produces comparative markdown table.
"""

import os
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Configuration combinations for ablation
CONFIGS = [
    {"name": "Baseline (Linear)", "ENABLE_STRUCTURED_OUTPUT": "False", "ENABLE_TABLE_TOOL": "False", "ENABLE_REACT": "False", "ENABLE_CLAIM_CHECKER": "False"},
    {"name": "+Structured Output", "ENABLE_STRUCTURED_OUTPUT": "True", "ENABLE_TABLE_TOOL": "False", "ENABLE_REACT": "False", "ENABLE_CLAIM_CHECKER": "False"},
    {"name": "+Table Tool", "ENABLE_STRUCTURED_OUTPUT": "True", "ENABLE_TABLE_TOOL": "True", "ENABLE_REACT": "False", "ENABLE_CLAIM_CHECKER": "False"},
    {"name": "+ReAct Loop", "ENABLE_STRUCTURED_OUTPUT": "True", "ENABLE_TABLE_TOOL": "True", "ENABLE_REACT": "True", "ENABLE_CLAIM_CHECKER": "False"},
    {"name": "+Claim Checker", "ENABLE_STRUCTURED_OUTPUT": "True", "ENABLE_TABLE_TOOL": "True", "ENABLE_REACT": "True", "ENABLE_CLAIM_CHECKER": "True"},
    {"name": "Full (All On)", "ENABLE_STRUCTURED_OUTPUT": "True", "ENABLE_TABLE_TOOL": "True", "ENABLE_REACT": "True", "ENABLE_CLAIM_CHECKER": "True"},
]

DATASET_FILE = Path("/home/tawil/Bureau/stage d'ete/dataset_eval.jsonl")
EVAL_SCRIPT = Path("/home/tawil/Bureau/stage d'ete/evaluate_extended.py")
WORK_DIR = Path("/home/tawil/Bureau/stage d'ete")
OUTPUT_DIR = Path("/home/tawil/Bureau/stage d'ete/ablation_results")

METRICS = [
    "faithfulness",
    "answer_relevancy", 
    "context_precision",
    "recall_at_5",
    "mrr",
    "ndcg_at_10",
    "citation_accuracy",
    "abstention_rate_correct",
    "routing_accuracy",
    "invalid_output_rate",
    "elapsed_sec",
    "tokens_estimate",
]


def run_evaluation(config: Dict[str, str], max_questions: int = 30) -> Dict[str, Any]:
    """Run evaluation with given config."""
    env = os.environ.copy()
    for k, v in config.items():
        if k != "name":
            env[k] = v
    
    # Unique results file per config
    results_file = OUTPUT_DIR / f"results_{config['name'].replace(' ', '_').replace('+', 'plus').replace('(', '').replace(')', '').lower()}.jsonl"
    env["RESULTS_FILE"] = str(results_file)
    
    cmd = ["python3", str(EVAL_SCRIPT), "--skip-llm-eval", "--max-questions", str(max_questions)]
    
    print(f"\n{'='*60}")
    print(f"Running: {config['name']}")
    print(f"{'='*60}")
    
    start = time.time()
    try:
        result = subprocess.run(
            cmd, 
            cwd=WORK_DIR, 
            env=env, 
            capture_output=True, 
            text=True, 
            timeout=600
        )
        elapsed = time.time() - start
        
        if result.returncode != 0:
            print(f"  ERROR (exit {result.returncode}): {result.stderr[-500:]}")
            return {"config": config['name'], "error": result.stderr[-500:], "elapsed": elapsed}
        
        # Parse results
        metrics = parse_results(results_file)
        metrics["elapsed_total"] = elapsed
        metrics["config"] = config['name']
        print(f"  Completed in {elapsed:.1f}s")
        return metrics
        
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 600s")
        return {"config": config['name'], "error": "TIMEOUT", "elapsed": 600}
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return {"config": config['name'], "error": str(e), "elapsed": time.time() - start}


def parse_results(results_file: Path) -> Dict[str, float]:
    """Parse evaluation results and compute averages."""
    if not results_file.exists():
        return {}
    
    results = []
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    
    if not results:
        return {}
    
    # Compute averages for each metric
    metrics = {}
    for m in METRICS:
        values = [r.get(m) for r in results if r.get(m) is not None]
        if values:
            metrics[m] = sum(values) / len(values)
        else:
            metrics[m] = None
    
    # Routing accuracy (special handling)
    routing_correct = sum(1 for r in results if r.get("routing_correct", False))
    metrics["routing_accuracy"] = routing_correct / len(results) if results else 0
    
    return metrics


def generate_markdown_table(all_results: List[Dict]) -> str:
    """Generate markdown comparison table."""
    lines = []
    lines.append("# Ablation Study Results")
    lines.append(f"*Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    def count_jsonl_lines(filepath: Path) -> int:
        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        json.loads(line)
                        count += 1
                    except json.JSONDecodeError:
                        pass
        return count

    lines.append(f"*Dataset: {count_jsonl_lines(DATASET_FILE)} questions (sampled)*")
    lines.append("")
    
    # Main metrics table
    lines.append("## Métriques Principales")
    lines.append("")
    
    # Header
    configs = [r["config"] for r in all_results if "config" in r]
    header = "| Métrique | " + " | ".join(configs) + " |"
    separator = "|----------|" + "|".join(["----------"] * len(configs)) + "|"
    lines.append(header)
    lines.append(separator)
    
    # Metrics rows (only those with data)
    display_metrics = [
        ("faithfulness", "Faithfulness"),
        ("answer_relevancy", "Answer Relevancy"),
        ("context_precision", "Context Precision"),
        ("recall_at_5", "Recall@5"),
        ("mrr", "MRR"),
        ("ndcg_at_10", "nDCG@10"),
        ("citation_accuracy", "Citation Accuracy"),
        ("abstention_rate_correct", "Abstention Correct"),
        ("routing_accuracy", "Routing Accuracy"),
        ("invalid_output_rate", "Invalid Output Rate"),
        ("elapsed_sec", "Latency (s)"),
        ("tokens_estimate", "Tokens/Query"),
    ]
    
    for key, label in display_metrics:
        row = f"| {label} |"
        for r in all_results:
            if "error" in r:
                row += " ERROR |"
            else:
                val = r.get(key)
                if val is not None:
                    if key in ["elapsed_sec", "tokens_estimate"]:
                        row += f" {val:.1f} |"
                    elif key == "invalid_output_rate":
                        row += f" {val:.3f} |"
                    else:
                        row += f" {val:.3f} |"
                else:
                    row += " N/A |"
        lines.append(row)
    
    lines.append("")
    
    # Delta table (vs baseline)
    lines.append("## Deltas vs Baseline")
    lines.append("")
    
    baseline = all_results[0] if all_results else {}
    if baseline and "error" not in baseline:
        header = "| Métrique | " + " | ".join([r["config"] for r in all_results[1:] if "error" not in r]) + " |"
        lines.append(header)
        lines.append(separator)
        
        for key, label in display_metrics:
            base_val = baseline.get(key)
            if base_val is None:
                continue
            row = f"| {label} |"
            for r in all_results[1:]:
                if "error" in r:
                    row += " ERROR |"
                else:
                    val = r.get(key)
                    if val is not None and base_val != 0:
                        if key == "elapsed_sec" or key == "tokens_estimate":
                            delta = val - base_val
                            row += f" {delta:+.1f} |"
                        elif key == "invalid_output_rate":
                            delta = val - base_val
                            row += f" {delta:+.3f} |"
                        else:
                            delta = (val - base_val) * 100  # percentage points
                            row += f" {delta:+.1f}pp |"
                    else:
                        row += " N/A |"
            lines.append(row)
    
    lines.append("")
    
    # Trajectory metrics (if available)
    lines.append("## Métriques de Trajectoire (ReAct)")
    lines.append("")
    lines.append("| Config | Avg Steps | Tool Choice Acc | Token Budget % | Time Budget % |")
    lines.append("|--------|-----------|-----------------|----------------|---------------|")
    for r in all_results:
        if "error" in r:
            continue
        # These would need to be collected during evaluation
        lines.append(f"| {r['config']} | N/A | N/A | N/A | N/A |")
    
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--max-questions", type=int, default=20, help="Max questions per config")
    parser.add_argument("--configs", nargs="+", help="Specific configs to run (by name)")
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    configs_to_run = CONFIGS
    if args.configs:
        configs_to_run = [c for c in CONFIGS if c["name"] in args.configs]
    
    print(f"Running ablation study with {len(configs_to_run)} configurations")
    print(f"Max questions per config: {args.max_questions}")
    
    all_results = []
    for config in configs_to_run:
        result = run_evaluation(config, max_questions=args.max_questions)
        all_results.append(result)
    
    # Generate markdown
    markdown = generate_markdown_table(all_results)
    
    output_file = OUTPUT_DIR / f"ablation_results_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    print(markdown)
    
    # Also save raw JSON
    json_file = OUTPUT_DIR / f"ablation_raw_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Raw data saved to: {json_file}")


if __name__ == "__main__":
    main()