#!/usr/bin/env python3
"""
Extended Evaluation Pipeline for AutoDrive Multi-Agent RAG System.
Adds retrieval metrics (Recall@5, MRR, nDCG@10), citation accuracy, abstention rate,
latency p95, tokens per query - on top of existing RAGAS metrics.
"""

import json
import argparse
import logging
import time
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from collections import defaultdict
import math

# LangChain imports for Ollama
from langchain_ollama import ChatOllama

# Local imports
import sys
sys.path.insert(0, "/home/tawil/Bureau/stage d'ete")
from core import process_question, get_or_create_session, close_resources
from hybrid_retrieval import retrieve_hybrid
from graph.nodes import superviseur
from utils.structured_output import reset_structured_output_stats, get_structured_output_stats

# Configuration
DATASET_FILE = Path("/home/tawil/Bureau/stage d'ete/dataset_eval.jsonl")
RESULTS_FILE = Path(os.getenv("RESULTS_FILE", "/home/tawil/Bureau/stage d'ete/resultats_eval_extended.jsonl"))
SUMMARY_FILE = Path("/home/tawil/Bureau/stage d'ete/eval_summary_baseline.md")

TARGETS = {
    "faithfulness": 0.87,
    "answer_relevancy": 0.82,
    "context_precision": 0.78,
    "recall_at_5": 0.80,
    "mrr": 0.70,
    "ndcg_at_10": 0.75,
    "citation_accuracy": 0.85,
    "abstention_rate_correct": 0.90,
    "routing_accuracy": 0.90,
    "invalid_output_rate": 0.02,  # Target: < 2% invalid structured outputs
}

LOG_FORMAT = "%(asctime)s [EVAL_EXT] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("evaluate_extended")

# Evaluation prompts for Ollama (same as evaluate_ragas.py)
FAITHFULNESS_PROMPT = """Tu es un évaluateur. Donne une note de 0 à 1 (un seul nombre) pour la fidélité de la réponse par rapport aux contextes fournis.

CONTEXTE:
{contexts}

RÉPONSE:
{answer}

RÈGLE: 1 = la réponse ne contient que des informations présentes dans le contexte. 0 = la réponse contient des hallucinations ou infos non dans le contexte.

NOTE (0.0 à 1.0):"""

ANSWER_RELEVANCY_PROMPT = """Tu es un évaluateur. Donne une note de 0 à 1 (un seul nombre) pour la pertinence de la réponse par rapport à la question.

QUESTION:
{question}

RÉPONSE:
{answer}

RÈGLE: 1 = la réponse répond directement et complètement à la question. 0 = la réponse est hors sujet ou incomplète.

NOTE (0.0 à 1.0):"""

CONTEXT_PRECISION_PROMPT = """Tu es un évaluateur. Donne une note de 0 à 1 (un seul nombre) pour la précision des contextes récupérés par rapport à la vérité terrain.

QUESTION:
{question}

CONTEXTES RÉCUPÉRÉS:
{contexts}

VÉRITÉ TERRAIN:
{ground_truth}

RÈGLE: 1 = les contextes contiennent exactement l'info nécessaire pour répondre. 0 = les contextes sont non pertinents.

NOTE (0.0 à 1.0):"""


def load_dataset(filepath: Path) -> List[Dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def load_existing_results(filepath: Path) -> Dict[str, Dict]:
    if not filepath.exists():
        return {}
    results = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    results[data["id"]] = data
                except json.JSONDecodeError:
                    pass
    return results


def save_result(filepath: Path, result: Dict):
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def extract_score(llm_response: str) -> Optional[float]:
    """Extract numeric score from LLM response."""
    matches = re.findall(r'0?\.\d+|1\.0|0|1', llm_response)
    if matches:
        try:
            score = float(matches[0])
            return max(0.0, min(1.0, score))
        except ValueError:
            pass
    return None


def evaluate_with_ollama(llm: ChatOllama, prompt: str) -> Optional[float]:
    """Evaluate using Ollama and extract score."""
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        return extract_score(content)
    except Exception as e:
        logger.warning(f"Ollama evaluation failed: {e}")
        return None


def compute_recall_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int = 5) -> float:
    """Compute Recall@k: proportion of expected passages found in top-k retrieved."""
    if not expected_ids:
        return 1.0  # No expected passages = vacuously satisfied
    retrieved_set = set(retrieved_ids[:k])
    expected_set = set(expected_ids)
    found = len(retrieved_set & expected_set)
    return found / len(expected_set)


def compute_mrr(retrieved_ids: List[str], expected_ids: List[str]) -> float:
    """Compute Mean Reciprocal Rank: 1/rank of first relevant passage."""
    if not expected_ids:
        return 1.0
    expected_set = set(expected_ids)
    for i, rid in enumerate(retrieved_ids):
        if rid in expected_set:
            return 1.0 / (i + 1)
    return 0.0


def compute_ndcg_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int = 10) -> float:
    """Compute nDCG@k assuming binary relevance (1 if in expected, 0 otherwise)."""
    if not expected_ids:
        return 1.0
    expected_set = set(expected_ids)
    
    # DCG
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        rel = 1.0 if rid in expected_set else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 2)
    
    # IDCG (ideal DCG)
    ideal_rels = [1.0] * min(len(expected_ids), k)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_citation_accuracy(response: str, retrieved_sources: List[str], retrieved_metadata: List[Dict]) -> float:
    """
    Compute citation accuracy: check if sources cited in response match retrieved sources.
    Heuristic: look for source-like patterns in response (file names, chunk IDs, etc.)
    """
    if not retrieved_sources:
        return 1.0 if "non trouvée" in response.lower() or "hors" in response.lower() else 0.5
    
    # Extract potential citations from response
    # Look for patterns like: [source], (source), source.jsonl, chunk_id patterns
    cited_sources = set()
    
    # Pattern 1: filenames like "politiques_contrats.jsonl", "chunks_contrats.jsonl"
    for src in retrieved_sources:
        if src and src != "inconnu":
            cited_sources.add(src.lower())
    
    # Pattern 2: document IDs like PC-0001, ASS-0001, etc.
    doc_id_pattern = r'\b([A-Z]{2,4}-\d{4})\b'
    for match in re.finditer(doc_id_pattern, response):
        cited_sources.add(match.group(1).lower())
    
    # Pattern 3: chunk IDs (8+ hex chars)
    chunk_pattern = r'\b([a-f0-9]{8,})\b'
    for match in re.finditer(chunk_pattern, response):
        cited_sources.add(match.group(1).lower())
    
    # Check overlap
    retrieved_lower = {s.lower() for s in retrieved_sources if s != "inconnu"}
    if not retrieved_lower:
        return 0.5
    
    # Also check metadata for document_id
    meta_sources = set()
    for m in retrieved_metadata:
        doc_id = m.get("document_id", "")
        if doc_id:
            meta_sources.add(doc_id.lower())
        src = m.get("source", "")
        if src:
            meta_sources.add(src.lower())
    
    all_retrieved = retrieved_lower | meta_sources
    if not all_retrieved:
        return 0.5
    
    # Compute precision of citations
    if cited_sources:
        overlap = len(cited_sources & all_retrieved)
        return overlap / max(len(cited_sources), 1)
    else:
        # No explicit citations - penalize unless abstention
        if any(w in response.lower() for w in ["non trouvée", "hors", "périmètre", "ne peux pas", "je ne sais"]):
            return 1.0
        return 0.3


def compute_abstention_correct(response: str, question_type: str, expected_agent: str, agent_used: str) -> bool:
    """
    Check if abstention is correct:
    - For sans_reponse: should refuse/abstain
    - For ambigue: should ask for clarification or give partial answer with caveat
    - For hors_sujet: should refuse
    """
    response_lower = response.lower()
    is_refusal = any(w in response_lower for w in [
        "non trouvée", "hors", "périmètre", "expertise", "ne peux pas", 
        "je ne sais", "information non", "pas dans la base", "précisez"
    ])
    
    if question_type == "sans_reponse":
        return is_refusal
    elif question_type == "hors_sujet":
        return is_refusal and (agent_used == "hors_sujet" or "hors" in response_lower)
    elif question_type == "ambigue":
        # For ambiguous, either clarification request or partial answer with caveat
        return is_refusal or "précisez" in response_lower or "dépend" in response_lower
    else:
        # For simple/chiffree/multi-hop: should NOT abstain
        return not is_refusal


def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~1.3 tokens per word for French."""
    words = len(text.split())
    return int(words * 1.3)


def evaluate_single_question(question_data: Dict, eval_llm: ChatOllama) -> Dict:
    qid = question_data["id"]
    question = question_data["question"]
    ground_truth = question_data["ground_truth"]
    expected_agent = question_data["agent_attendu"]
    category = question_data["categorie"]
    expected_passages = question_data.get("passages_attendus", [])
    question_type = question_data.get("type_question", "simple")
    
    logger.info(f"Evaluating {qid}: {question[:60]}...")
    
    # Reset structured output stats for this question
    reset_structured_output_stats()
    
    session_id, history = get_or_create_session()
    
    start_time = time.time()
    
    # Invoke the full graph
    try:
        result = process_question(question, session_id, history)
        response = result.get("reponse", "")
        sources = result.get("sources", [])
        agent_used = result.get("agent_utilise", "unknown")
        elapsed = time.time() - start_time
    except Exception as e:
        logger.error(f"Error processing question {qid}: {e}")
        return {
            "id": qid, "question": question, "categorie": category,
            "agent_attendu": expected_agent, "agent_utilise": "error",
            "ground_truth": ground_truth, "reponse": "", "contexts": [],
            "error": str(e), "faithfulness": None, "answer_relevancy": None,
            "context_precision": None, "recall_at_5": 0.0, "mrr": 0.0,
            "ndcg_at_10": 0.0, "citation_accuracy": 0.0,
            "abstention_correct": False, "routing_correct": False,
            "elapsed_sec": time.time() - start_time, "tokens_estimate": 0,
            "timestamp": datetime.now().isoformat(),
        }
    
    # Get retrieved contexts AND their metadata/chunk_ids for retrieval metrics
    try:
        state = {"question": question}
        agents_result = superviseur(state)
        agents = agents_result.get("agents", [])
        agent_category = agents[0] if agents else "contrats"
        
        docs, metadatas = retrieve_hybrid(question, agent_category, top_k_final=10)
        contexts = docs
        
        # Extract chunk_ids from metadata (if available) or use document_id
        retrieved_chunk_ids = []
        for m in metadatas:
            chunk_id = m.get("chunk_id", "")
            doc_id = m.get("document_id", "")
            if chunk_id:
                retrieved_chunk_ids.append(chunk_id)
            elif doc_id:
                retrieved_chunk_ids.append(doc_id)
        
        # Also get source filenames
        retrieved_sources = []
        for m in metadatas:
            src = m.get("source", "inconnu")
            retrieved_sources.append(src)
            
    except Exception as e:
        logger.warning(f"Could not retrieve contexts for {qid}: {e}")
        contexts = []
        retrieved_chunk_ids = []
        retrieved_sources = []
        metadatas = []
        agent_category = "contrats"
    
    # Evaluate with Ollama (RAGAS metrics)
    contexts_str = "\n\n---\n\n".join(contexts[:5]) if contexts else "Aucun contexte récupéré"
    
    # Faithfulness
    faith_prompt = FAITHFULNESS_PROMPT.format(contexts=contexts_str, answer=response)
    faithfulness_score = evaluate_with_ollama(eval_llm, faith_prompt)
    
    # Answer Relevancy
    rel_prompt = ANSWER_RELEVANCY_PROMPT.format(question=question, answer=response)
    relevancy_score = evaluate_with_ollama(eval_llm, rel_prompt)
    
    # Context Precision
    prec_prompt = CONTEXT_PRECISION_PROMPT.format(
        question=question, contexts=contexts_str, ground_truth=ground_truth
    )
    precision_score = evaluate_with_ollama(eval_llm, prec_prompt)
    
    # Routing accuracy
    routing_correct = (agent_used == expected_agent) or \
                     (expected_agent == "hors_sujet" and agent_used == "hors_sujet")
    
    # Off-topic refusal accuracy
    off_topic_correct = False
    if expected_agent == "hors_sujet":
        off_topic_correct = any(w in response.lower() for w in ["hors", "périmètre", "expertise", "ne peux pas"])
    
    # NEW: Retrieval metrics
    recall_at_5 = compute_recall_at_k(retrieved_chunk_ids, expected_passages, k=5)
    mrr = compute_mrr(retrieved_chunk_ids, expected_passages)
    ndcg_at_10 = compute_ndcg_at_k(retrieved_chunk_ids, expected_passages, k=10)
    
    # NEW: Citation accuracy
    citation_acc = compute_citation_accuracy(response, retrieved_sources, metadatas)
    
    # NEW: Abstention correctness
    abstention_correct = compute_abstention_correct(response, question_type, expected_agent, agent_used)
    
    # NEW: Token estimation
    tokens_estimate = estimate_tokens(response) + estimate_tokens(contexts_str)
    
    # NEW: Structured output invalid rate (per question)
    stats = get_structured_output_stats()
    total_calls = stats["total_calls"]
    invalid_rate = stats["final_failures"] / max(total_calls, 1)
    
    eval_result = {
        "id": qid,
        "question": question,
        "categorie": category,
        "agent_attendu": expected_agent,
        "agent_utilise": agent_used,
        "ground_truth": ground_truth,
        "reponse": response[:2000],
        "contexts": contexts[:5],
        "retrieved_chunk_ids": retrieved_chunk_ids[:10],
        "retrieved_sources": retrieved_sources[:10],
        "expected_passages": expected_passages,
        "question_type": question_type,
        "faithfulness": faithfulness_score,
        "answer_relevancy": relevancy_score,
        "context_precision": precision_score,
        "recall_at_5": recall_at_5,
        "mrr": mrr,
        "ndcg_at_10": ndcg_at_10,
        "citation_accuracy": citation_acc,
        "abstention_correct": abstention_correct,
        "routing_correct": routing_correct,
        "off_topic_correct": off_topic_correct,
        "elapsed_sec": round(elapsed, 2),
        "tokens_estimate": tokens_estimate,
        "invalid_output_rate": round(invalid_rate, 4),
        "structured_output_stats": stats,
        "timestamp": datetime.now().isoformat(),
    }
    
    return eval_result


def print_summary(results: List[Dict]):
    if not results:
        print("No results to summarize.")
        return
    
    print("\n" + "="*100)
    print("EXTENDED EVALUATION SUMMARY (Ollama-based + Retrieval Metrics)")
    print("="*100)
    
    # Filter valid scores
    metrics = ["faithfulness", "answer_relevancy", "context_precision", 
               "recall_at_5", "mrr", "ndcg_at_10", "citation_accuracy", "invalid_output_rate"]
    
    print(f"\nGlobal Metrics:")
    for metric_name in metrics:
        valid = [r[metric_name] for r in results if r.get(metric_name) is not None]
        if valid:
            avg = sum(valid)/len(valid)
            target = TARGETS.get(metric_name, 0)
            if metric_name == "invalid_output_rate":
                status = '✓' if avg <= target else '✗'
            else:
                status = '✓' if avg >= target else '✗'
            print(f"  {metric_name:25s}: {avg:.4f} (target: {target}) {status}")
    
    # Abstention rate
    abstention_valid = [r["abstention_correct"] for r in results if "abstention_correct" in r]
    if abstention_valid:
        avg = sum(abstention_valid)/len(abstention_valid)
        target = TARGETS.get("abstention_rate_correct", 0)
        status = '✓' if avg >= target else '✗'
        print(f"  {'abstention_rate_correct':25s}: {avg:.4f} (target: {target}) {status}")
    
    total = len(results)
    routing_correct = sum(1 for r in results if r.get("routing_correct", False))
    target = TARGETS.get("routing_accuracy", 0)
    status = '✓' if routing_correct/total >= target else '✗'
    print(f"  {'routing_accuracy':25s}: {routing_correct}/{total} = {routing_correct/total*100:.1f}% (target: {target*100:.0f}%) {status}")
    
    off_topic = [r for r in results if r["agent_attendu"] == "hors_sujet"]
    off_topic_correct = sum(1 for r in off_topic if r.get("off_topic_correct", False))
    if len(off_topic) > 0:
        print(f"  {'off_topic_refusal':25s}: {off_topic_correct}/{len(off_topic)} = {off_topic_correct/len(off_topic)*100:.1f}%")
    
    # Per type breakdown
    print(f"\nPer Question Type:")
    types = set(r.get("question_type", "simple") for r in results)
    for qtype in sorted(types):
        type_results = [r for r in results if r.get("question_type") == qtype]
        type_faith = [r["faithfulness"] for r in type_results if r.get("faithfulness") is not None]
        type_recall = [r["recall_at_5"] for r in type_results if r.get("recall_at_5") is not None]
        type_mrr = [r["mrr"] for r in type_results if r.get("mrr") is not None]
        type_ndcg = [r["ndcg_at_10"] for r in type_results if r.get("ndcg_at_10") is not None]
        type_cite = [r["citation_accuracy"] for r in type_results if r.get("citation_accuracy") is not None]
        type_abs = [r["abstention_correct"] for r in type_results if "abstention_correct" in r]
        type_routing = sum(1 for r in type_results if r.get("routing_correct", False))
        
        if type_faith:
            print(f"  {qtype:15s} | F:{sum(type_faith)/len(type_faith):.3f} R@5:{sum(type_recall)/len(type_recall):.3f} "
                  f"MRR:{sum(type_mrr)/len(type_mrr):.3f} nDCG:{sum(type_ndcg)/len(type_ndcg):.3f} "
                  f"Cite:{sum(type_cite)/len(type_cite):.3f} Abs:{sum(type_abs)/len(type_abs):.3f} | Route:{type_routing}/{len(type_results)} | N:{len(type_results)}")
    
    # Per category breakdown
    print(f"\nPer Category:")
    categories = set(r["categorie"] for r in results)
    for cat in sorted(categories):
        cat_results = [r for r in results if r["categorie"] == cat]
        cat_faith = [r["faithfulness"] for r in cat_results if r.get("faithfulness") is not None]
        cat_rel = [r["answer_relevancy"] for r in cat_results if r.get("answer_relevancy") is not None]
        cat_prec = [r["context_precision"] for r in cat_results if r.get("context_precision") is not None]
        cat_recall = [r["recall_at_5"] for r in cat_results if r.get("recall_at_5") is not None]
        cat_routing = sum(1 for r in cat_results if r.get("routing_correct", False))
        
        if cat_faith:
            print(f"  {cat:15s} | F:{sum(cat_faith)/len(cat_faith):.3f} R:{sum(cat_rel)/len(cat_rel):.3f} "
                  f"P:{sum(cat_prec)/len(cat_prec):.3f} R@5:{sum(cat_recall)/len(cat_recall):.3f} | Route:{cat_routing}/{len(cat_results)} | N:{len(cat_results)}")
    
    # Latency stats
    latencies = [r.get("elapsed_sec", 0) for r in results if r.get("elapsed_sec")]
    if latencies:
        latencies.sort()
        p50 = latencies[len(latencies)//2]
        p95 = latencies[int(len(latencies)*0.95)]
        avg_time = sum(latencies) / len(latencies)
        print(f"\nLatency: avg={avg_time:.1f}s, p50={p50:.1f}s, p95={p95:.1f}s")
    
    # Token stats
    tokens = [r.get("tokens_estimate", 0) for r in results if r.get("tokens_estimate")]
    if tokens:
        avg_tokens = sum(tokens) / len(tokens)
        print(f"Avg tokens/query (est.): {avg_tokens:.0f}")


def generate_markdown_summary(results: List[Dict], output_file: Path):
    """Generate markdown summary table for the memory."""
    lines = []
    lines.append("# AutoDrive RAG - Baseline Evaluation Summary")
    lines.append(f"*Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append(f"*Architecture: Linear pipeline (reformulateur → supervisor → agent RAG)*")
    lines.append("")
    
    # Global metrics table
    lines.append("## Métriques Globales")
    lines.append("")
    lines.append("| Métrique | Valeur | Cible | Statut |")
    lines.append("|----------|--------|-------|--------|")
    
    metrics = ["faithfulness", "answer_relevancy", "context_precision", 
               "recall_at_5", "mrr", "ndcg_at_10", "citation_accuracy", "invalid_output_rate"]
    
    for metric_name in metrics:
        valid = [r[metric_name] for r in results if r.get(metric_name) is not None]
        if valid:
            avg = sum(valid)/len(valid)
            target = TARGETS.get(metric_name, 0)
            if metric_name == "invalid_output_rate":
                status = "✅" if avg <= target else "❌"
            else:
                status = "✅" if avg >= target else "❌"
            lines.append(f"| {metric_name} | {avg:.4f} | {target} | {status} |")
    
    abstention_valid = [r["abstention_correct"] for r in results if "abstention_correct" in r]
    if abstention_valid:
        avg = sum(abstention_valid)/len(abstention_valid)
        target = TARGETS.get("abstention_rate_correct", 0)
        status = "✅" if avg >= target else "❌"
        lines.append(f"| abstention_rate_correct | {avg:.4f} | {target} | {status} |")
    
    total = len(results)
    routing_correct = sum(1 for r in results if r.get("routing_correct", False))
    target = TARGETS.get("routing_accuracy", 0)
    status = "✅" if routing_correct/total >= target else "❌"
    lines.append(f"| routing_accuracy | {routing_correct/total:.4f} | {target} | {status} |")
    
    lines.append("")
    
    # Latency
    latencies = [r.get("elapsed_sec", 0) for r in results if r.get("elapsed_sec")]
    if latencies:
        latencies.sort()
        p50 = latencies[len(latencies)//2]
        p95 = latencies[int(len(latencies)*0.95)]
        avg_time = sum(latencies) / len(latencies)
        lines.append("## Latence")
        lines.append("")
        lines.append(f"- Moyenne: {avg_time:.1f}s")
        lines.append(f"- p50: {p50:.1f}s")
        lines.append(f"- p95: {p95:.1f}s")
        lines.append("")
    
    # Tokens
    tokens = [r.get("tokens_estimate", 0) for r in results if r.get("tokens_estimate")]
    if tokens:
        avg_tokens = sum(tokens) / len(tokens)
        lines.append("## Tokens par requête (estimation)")
        lines.append("")
        lines.append(f"- Moyenne: {avg_tokens:.0f} tokens")
        lines.append("")
    
    # Per type
    lines.append("## Par Type de Question")
    lines.append("")
    lines.append("| Type | Faith. | Recall@5 | MRR | nDCG@10 | Citation | Abstention | Routing | N |")
    lines.append("|------|--------|----------|-----|---------|----------|------------|---------|---|")
    
    types = set(r.get("question_type", "simple") for r in results)
    for qtype in sorted(types):
        type_results = [r for r in results if r.get("question_type") == qtype]
        type_faith = [r["faithfulness"] for r in type_results if r.get("faithfulness") is not None]
        type_recall = [r["recall_at_5"] for r in type_results if r.get("recall_at_5") is not None]
        type_mrr = [r["mrr"] for r in type_results if r.get("mrr") is not None]
        type_ndcg = [r["ndcg_at_10"] for r in type_results if r.get("ndcg_at_10") is not None]
        type_cite = [r["citation_accuracy"] for r in type_results if r.get("citation_accuracy") is not None]
        type_abs = [r["abstention_correct"] for r in type_results if "abstention_correct" in r]
        type_routing = sum(1 for r in type_results if r.get("routing_correct", False))
        
        if type_faith:
            lines.append(f"| {qtype} | {sum(type_faith)/len(type_faith):.3f} | {sum(type_recall)/len(type_recall):.3f} | "
                        f"{sum(type_mrr)/len(type_mrr):.3f} | {sum(type_ndcg)/len(type_ndcg):.3f} | "
                        f"{sum(type_cite)/len(type_cite):.3f} | {sum(type_abs)/len(type_abs):.3f} | "
                        f"{type_routing}/{len(type_results)} | {len(type_results)} |")
    
    lines.append("")
    
    # Per category
    lines.append("## Par Catégorie")
    lines.append("")
    lines.append("| Catégorie | Faith. | Relev. | Prec. | Recall@5 | Routing | N |")
    lines.append("|-----------|--------|--------|-------|----------|---------|---|")
    
    categories = set(r["categorie"] for r in results)
    for cat in sorted(categories):
        cat_results = [r for r in results if r["categorie"] == cat]
        cat_faith = [r["faithfulness"] for r in cat_results if r.get("faithfulness") is not None]
        cat_rel = [r["answer_relevancy"] for r in cat_results if r.get("answer_relevancy") is not None]
        cat_prec = [r["context_precision"] for r in cat_results if r.get("context_precision") is not None]
        cat_recall = [r["recall_at_5"] for r in cat_results if r.get("recall_at_5") is not None]
        cat_routing = sum(1 for r in cat_results if r.get("routing_correct", False))
        
        if cat_faith:
            lines.append(f"| {cat} | {sum(cat_faith)/len(cat_faith):.3f} | {sum(cat_rel)/len(cat_rel):.3f} | "
                        f"{sum(cat_prec)/len(cat_prec):.3f} | {sum(cat_recall)/len(cat_recall):.3f} | "
                        f"{cat_routing}/{len(cat_results)} | {len(cat_results)} |")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    logger.info(f"Markdown summary saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Extended Evaluation for AutoDrive")
    parser.add_argument("--max-questions", type=int, default=None, help="Max questions to evaluate")
    parser.add_argument("--skip-llm-eval", action="store_true", help="Skip LLM-based evaluation (faster)")
    args = parser.parse_args()
    
    logger.info("Starting extended evaluation...")
    
    dataset = load_dataset(DATASET_FILE)
    logger.info(f"Loaded {len(dataset)} questions")
    
    existing = load_existing_results(RESULTS_FILE)
    logger.info(f"Found {len(existing)} existing results")
    
    to_evaluate = [q for q in dataset if q["id"] not in existing]
    if args.max_questions:
        to_evaluate = to_evaluate[:args.max_questions]
    
    logger.info(f"Will evaluate {len(to_evaluate)} questions")
    
    if not to_evaluate:
        logger.info("All questions already evaluated.")
        all_results = list(existing.values())
        print_summary(all_results)
        generate_markdown_summary(all_results, SUMMARY_FILE)
        return
    
    # Initialize Ollama for evaluation
    if not args.skip_llm_eval:
        eval_llm = ChatOllama(
            model="mistral",
            temperature=0,
            base_url="http://localhost:11434",
        )
    else:
        eval_llm = None
    
    all_results = list(existing.values())
    
    for i, question_data in enumerate(to_evaluate):
        logger.info(f"Progress: {i+1}/{len(to_evaluate)}")
        
        try:
            if args.skip_llm_eval:
                session_id, history = get_or_create_session()
                result = process_question(question_data["question"], session_id, history)
                
                eval_result = {
                    "id": question_data["id"],
                    "question": question_data["question"],
                    "categorie": question_data["categorie"],
                    "agent_attendu": question_data["agent_attendu"],
                    "agent_utilise": result.get("agent_utilise", "unknown"),
                    "ground_truth": question_data["ground_truth"],
                    "reponse": result.get("reponse", "")[:2000],
                    "contexts": [],
                    "retrieved_chunk_ids": [],
                    "retrieved_sources": [],
                    "expected_passages": question_data.get("passages_attendus", []),
                    "question_type": question_data.get("type_question", "simple"),
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "context_precision": None,
                    "recall_at_5": 0.0,
                    "mrr": 0.0,
                    "ndcg_at_10": 0.0,
                    "citation_accuracy": 0.0,
                    "abstention_correct": False,
                    "routing_correct": result.get("agent_utilise") == question_data["agent_attendu"],
                    "off_topic_correct": False,
                    "elapsed_sec": 0,
                    "tokens_estimate": 0,
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                eval_result = evaluate_single_question(question_data, eval_llm)
            
            all_results.append(eval_result)
            save_result(RESULTS_FILE, eval_result)
            
        except Exception as e:
            logger.error(f"Failed to evaluate {question_data['id']}: {e}")
            continue
    
    print_summary(all_results)
    generate_markdown_summary(all_results, SUMMARY_FILE)
    close_resources()
    logger.info("Extended evaluation complete!")


if __name__ == "__main__":
    main()