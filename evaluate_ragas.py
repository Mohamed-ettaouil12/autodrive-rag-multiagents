#!/usr/bin/env python3
"""
RAGAS-like Evaluation Pipeline for AutoDrive Multi-Agent RAG System.
Uses Ollama (mistral) directly for evaluation metrics.
"""

import json
import argparse
import logging
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# LangChain imports for Ollama
from langchain_ollama import ChatOllama

# Local imports
import sys
sys.path.insert(0, "/home/tawil/Bureau/stage d'ete")
from core import process_question, get_or_create_session, close_resources
from hybrid_retrieval import retrieve_hybrid
from graph.nodes import superviseur

# Configuration
DATASET_FILE = Path("/home/tawil/Bureau/stage d'ete/dataset_eval.jsonl")
RESULTS_FILE = Path("/home/tawil/Bureau/stage d'ete/resultats_eval.jsonl")

# Target metrics
TARGETS = {
    "faithfulness": 0.87,
    "answer_relevancy": 0.82,
    "context_precision": 0.78,
}

LOG_FORMAT = "%(asctime)s [EVAL] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("evaluate_ragas")

# Evaluation prompts for Ollama
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
    import re
    # Find first float in response
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

def evaluate_single_question(question_data: Dict, eval_llm: ChatOllama) -> Dict:
    qid = question_data["id"]
    question = question_data["question"]
    ground_truth = question_data["ground_truth"]
    expected_agent = question_data["agent_attendu"]
    category = question_data["categorie"]
    
    logger.info(f"Evaluating {qid}: {question[:60]}...")
    
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
            "id": qid,
            "question": question,
            "categorie": category,
            "agent_attendu": expected_agent,
            "agent_utilise": "error",
            "ground_truth": ground_truth,
            "reponse": "",
            "contexts": [],
            "error": str(e),
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "routing_correct": False,
            "elapsed_sec": time.time() - start_time,
        }
    
    # Get retrieved contexts
    try:
        state = {"question": question}
        agents_result = superviseur(state)
        agents = agents_result.get("agents", [])
        agent_category = agents[0] if agents else "contrats"
        
        docs, metadatas = retrieve_hybrid(question, agent_category, top_k_final=5)
        contexts = docs
    except Exception as e:
        logger.warning(f"Could not retrieve contexts for {qid}: {e}")
        contexts = []
        agent_category = "contrats"
    
    # Evaluate with Ollama
    contexts_str = "\n\n---\n\n".join(contexts[:5]) if contexts else "Aucun contexte récupéré"
    
    # Faithfulness
    faith_prompt = FAITHFULNESS_PROMPT.format(contexts=contexts_str, answer=response)
    faithfulness_score = evaluate_with_ollama(eval_llm, faith_prompt)
    
    # Answer Relevancy
    rel_prompt = ANSWER_RELEVANCY_PROMPT.format(question=question, answer=response)
    relevancy_score = evaluate_with_ollama(eval_llm, rel_prompt)
    
    # Context Precision
    prec_prompt = CONTEXT_PRECISION_PROMPT.format(
        question=question, 
        contexts=contexts_str, 
        ground_truth=ground_truth
    )
    precision_score = evaluate_with_ollama(eval_llm, prec_prompt)
    
    # Routing accuracy
    routing_correct = (agent_used == expected_agent) or \
                     (expected_agent == "hors_sujet" and agent_used == "hors_sujet")
    
    # Off-topic refusal accuracy
    off_topic_correct = False
    if expected_agent == "hors_sujet":
        off_topic_correct = any(w in response.lower() for w in ["hors", "périmètre", "expertise", "ne peux pas"])
    
    eval_result = {
        "id": qid,
        "question": question,
        "categorie": category,
        "agent_attendu": expected_agent,
        "agent_utilise": agent_used,
        "ground_truth": ground_truth,
        "reponse": response[:1000],
        "contexts": contexts[:5],
        "faithfulness": faithfulness_score,
        "answer_relevancy": relevancy_score,
        "context_precision": precision_score,
        "routing_correct": routing_correct,
        "off_topic_correct": off_topic_correct,
        "elapsed_sec": round(time.time() - start_time, 2),
        "timestamp": datetime.now().isoformat(),
    }
    
    return eval_result

def print_summary(results: List[Dict]):
    if not results:
        print("No results to summarize.")
        return
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY (Ollama-based)")
    print("="*80)
    
    valid_faith = [r["faithfulness"] for r in results if r["faithfulness"] is not None]
    valid_rel = [r["answer_relevancy"] for r in results if r["answer_relevancy"] is not None]
    valid_prec = [r["context_precision"] for r in results if r["context_precision"] is not None]
    
    print(f"\nGlobal Metrics:")
    if valid_faith:
        avg = sum(valid_faith)/len(valid_faith)
        print(f"  Faithfulness:       {avg:.4f} (target: {TARGETS['faithfulness']}) {'✓' if avg >= TARGETS['faithfulness'] else '✗'}")
    if valid_rel:
        avg = sum(valid_rel)/len(valid_rel)
        print(f"  Answer Relevancy:   {avg:.4f} (target: {TARGETS['answer_relevancy']}) {'✓' if avg >= TARGETS['answer_relevancy'] else '✗'}")
    if valid_prec:
        avg = sum(valid_prec)/len(valid_prec)
        print(f"  Context Precision:  {avg:.4f} (target: {TARGETS['context_precision']}) {'✓' if avg >= TARGETS['context_precision'] else '✗'}")
    
    total = len(results)
    routing_correct = sum(1 for r in results if r["routing_correct"])
    off_topic = [r for r in results if r["agent_attendu"] == "hors_sujet"]
    off_topic_correct = sum(1 for r in off_topic if r["off_topic_correct"])
    
    print(f"\nRouting Accuracy: {routing_correct}/{total} = {routing_correct/total*100:.1f}%")
    if len(off_topic) > 0:
        print(f"Off-topic Refusal: {off_topic_correct}/{len(off_topic)} = {off_topic_correct/len(off_topic)*100:.1f}%")
    else:
        print(f"Off-topic Refusal: N/A (no off-topic questions in this run)")
    
    print(f"\nPer Category:")
    categories = set(r["categorie"] for r in results)
    for cat in sorted(categories):
        cat_results = [r for r in results if r["categorie"] == cat]
        cat_faith = [r["faithfulness"] for r in cat_results if r["faithfulness"] is not None]
        cat_rel = [r["answer_relevancy"] for r in cat_results if r["answer_relevancy"] is not None]
        cat_prec = [r["context_precision"] for r in cat_results if r["context_precision"] is not None]
        cat_routing = sum(1 for r in cat_results if r["routing_correct"])
        
        if cat_faith:
            print(f"  {cat:15s} | F:{sum(cat_faith)/len(cat_faith):.3f} R:{sum(cat_rel)/len(cat_rel):.3f} P:{sum(cat_prec)/len(cat_prec):.3f} | Route:{cat_routing}/{len(cat_results)} | N:{len(cat_results)}")
    
    avg_time = sum(r["elapsed_sec"] for r in results) / len(results)
    print(f"\nAvg response time: {avg_time:.1f}s")

def main():
    parser = argparse.ArgumentParser(description="Evaluation for AutoDrive")
    parser.add_argument("--max-questions", type=int, default=None, help="Max questions to evaluate")
    parser.add_argument("--skip-llm-eval", action="store_true", help="Skip LLM-based evaluation (faster)")
    args = parser.parse_args()
    
    logger.info("Starting evaluation...")
    
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
                    "reponse": result.get("reponse", "")[:1000],
                    "contexts": [],
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "context_precision": None,
                    "routing_correct": result.get("agent_utilise") == question_data["agent_attendu"],
                    "off_topic_correct": False,
                    "elapsed_sec": 0,
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
    close_resources()
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()