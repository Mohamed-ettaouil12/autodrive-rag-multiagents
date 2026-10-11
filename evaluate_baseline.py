#!/usr/bin/env python3
"""
Baseline Evaluation - Simple RAG without multi-agents, graph, or hybrid retrieval.
Uses only vector search across all collections concatenated.
"""

import json
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer
import chromadb

import sys
sys.path.insert(0, "/home/tawil/Bureau/stage d'ete")
from core import get_or_create_session, close_resources

DATASET_FILE = Path("/home/tawil/Bureau/stage d'ete/dataset_eval.jsonl")
BASELINE_RESULTS_FILE = Path("/home/tawil/Bureau/stage d'ete/resultats_baseline.jsonl")
COMPARISON_FILE = Path("/home/tawil/Bureau/stage d'ete/comparaison_baseline_vs_multiagents.md")

TARGETS = {
    "faithfulness": 0.87,
    "answer_relevancy": 0.82,
    "context_precision": 0.78,
}

LOG_FORMAT = "%(asctime)s [BASELINE] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("evaluate_baseline")

# All collections to search
ALL_COLLECTIONS = ["politiques_contrats", "assurances", "faq", "vehicules", "maintenance"]

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
    import re
    matches = re.findall(r'0?\.\d+|1\.0|0|1', llm_response)
    if matches:
        try:
            score = float(matches[0])
            return max(0.0, min(1.0, score))
        except ValueError:
            pass
    return None

class BaselineRAG:
    """Simple RAG: vector search across all collections."""
    
    def __init__(self):
        logger.info("Initializing Baseline RAG...")
        self.embedding_model = SentenceTransformer("BAAI/bge-m3", device="cpu")
        self.chroma_client = chromadb.PersistentClient(
            path="/home/tawil/Bureau/stage d'ete/preprocessing/output/chroma_db"
        )
        self.eval_llm = ChatOllama(model="mistral", temperature=0, base_url="http://localhost:11434")
        logger.info("Baseline RAG initialized")
    
    def search_all(self, question: str, top_k: int = 5) -> List[Dict]:
        """Search across all collections and merge results."""
        query_embedding = self.embedding_model.encode(
            question, normalize_embeddings=True, show_progress_bar=False
        ).tolist()
        
        all_results = []
        
        for collection_name in ALL_COLLECTIONS:
            try:
                collection = self.chroma_client.get_collection(collection_name)
                result = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, 10),
                    include=["documents", "metadatas", "distances"]
                )
                
                docs = result["documents"][0] if result["documents"] else []
                metas = result["metadatas"][0] if result["metadatas"] else []
                distances = result["distances"][0] if result["distances"] else []
                
                for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
                    all_results.append({
                        "text": doc,
                        "metadata": meta,
                        "score": 1.0 - dist,
                        "collection": collection_name,
                        "rank": i + 1,
                    })
            except Exception as e:
                logger.warning(f"Error searching {collection_name}: {e}")
        
        # Sort by score and take top_k
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]
    
    def generate_answer(self, question: str, contexts: List[str]) -> str:
        """Generate answer using Ollama with retrieved contexts."""
        if not contexts:
            return "Information non trouvée dans la base documentaire."
        
        context_str = "\n\n".join(contexts[:5])
        
        prompt = f"""Tu es un assistant AutoDrive Maroc. Réponds à la question en utilisant uniquement le contexte fourni.

CONTEXTE:
{context_str}

QUESTION:
{question}

RÉPONSE:"""
        
        try:
            response = self.eval_llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "Erreur lors de la génération de la réponse."

def evaluate_with_ollama(llm: ChatOllama, prompt: str) -> Optional[float]:
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        import re
        matches = re.findall(r'0?\.\d+|1\.0|0|1', content)
        if matches:
            score = float(matches[0])
            return max(0.0, min(1.0, score))
    except Exception as e:
        logger.warning(f"Ollama evaluation failed: {e}")
    return None

def evaluate_single_question(question_data: Dict, baseline: BaselineRAG) -> Dict:
    qid = question_data["id"]
    question = question_data["question"]
    ground_truth = question_data["ground_truth"]
    expected_agent = question_data["agent_attendu"]
    category = question_data["categorie"]
    
    logger.info(f"Evaluating baseline {qid}: {question[:60]}...")
    
    session_id, history = get_or_create_session()
    start_time = time.time()
    
    # Baseline: search all collections
    results = baseline.search_all(question, top_k=5)
    contexts = [r["text"] for r in results]
    
    # Generate answer
    answer = baseline.generate_answer(question, contexts)
    
    # No routing in baseline - agent is always "baseline"
    agent_used = "baseline"
    
    elapsed = time.time() - start_time
    
    # Evaluate with Ollama
    contexts_str = "\n\n---\n\n".join(contexts[:5]) if contexts else "Aucun contexte"
    
    # Faithfulness
    faith_prompt = FAITHFULNESS_PROMPT.format(contexts=contexts_str, answer=answer)
    faithfulness_score = evaluate_with_ollama(baseline.eval_llm, faith_prompt)
    
    # Answer Relevancy
    rel_prompt = ANSWER_RELEVANCY_PROMPT.format(question=question, answer=answer)
    relevancy_score = evaluate_with_ollama(baseline.eval_llm, rel_prompt)
    
    # Context Precision
    prec_prompt = CONTEXT_PRECISION_PROMPT.format(
        question=question, contexts=contexts_str, ground_truth=ground_truth
    )
    precision_score = evaluate_with_ollama(baseline.eval_llm, prec_prompt)
    
    # Routing accuracy (baseline has no routing)
    routing_correct = (agent_used == expected_agent) or \
                     (expected_agent == "hors_sujet" and "hors" in answer.lower())
    
    off_topic_correct = False
    if expected_agent == "hors_sujet":
        off_topic_correct = any(w in answer.lower() for w in ["hors", "périmètre", "expertise", "ne peux pas"])
    
    return {
        "id": qid,
        "question": question,
        "categorie": category,
        "agent_attendu": expected_agent,
        "agent_utilise": agent_used,
        "ground_truth": ground_truth,
        "reponse": answer[:1000],
        "contexts": contexts[:5],
        "faithfulness": faithfulness_score,
        "answer_relevancy": relevancy_score,
        "context_precision": precision_score,
        "routing_correct": routing_correct,
        "off_topic_correct": off_topic_correct,
        "elapsed_sec": round(time.time() - start_time, 2),
        "timestamp": datetime.now().isoformat(),
    }

def load_multiagent_results(filepath: Path) -> List[Dict]:
    """Load multi-agent results for comparison."""
    if not filepath.exists():
        return []
    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return results

def generate_comparison(baseline_results: List[Dict], multiagent_results: List[Dict]) -> str:
    """Generate markdown comparison table."""
    
    # Create lookup for multiagent results by id
    multiagent_by_id = {r["id"]: r for r in multiagent_results}
    
    lines = []
    lines.append("# Comparaison Baseline vs Multi-Agents")
    lines.append("")
    lines.append(f"*Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")
    
    # Global comparison
    lines.append("## Métriques Globales")
    lines.append("")
    lines.append("| Métrique | Baseline | Multi-Agents | Delta | Cible |")
    lines.append("|----------|----------|--------------|-------|-------|")
    
    # Compute global metrics
    for metric_name, target in TARGETS.items():
        base_vals = [r.get(metric_name) for r in baseline_results if r.get(metric_name) is not None]
        multi_vals = [r.get(metric_name) for r in multiagent_results if r.get(metric_name) is not None]
        
        base_avg = sum(base_vals)/len(base_vals) if base_vals else 0
        multi_avg = sum(multi_vals)/len(multi_vals) if multi_vals else 0
        delta = multi_avg - base_avg
        
        lines.append(f"| {metric_name} | {base_avg:.4f} | {multi_avg:.4f} | {delta:+.4f} | {target} |")
    
    # Routing accuracy
    base_routing = sum(1 for r in baseline_results if r.get("routing_correct", False)) / len(baseline_results) * 100
    multi_routing = sum(1 for r in multiagent_results if r.get("routing_correct", False)) / len(multiagent_results) * 100
    lines.append(f"| routing_accuracy | {base_routing:.1f}% | {multi_routing:.1f}% | {multi_routing-base_routing:+.1f}% | - |")
    
    # Avg time
    base_time = sum(r["elapsed_sec"] for r in baseline_results) / len(baseline_results)
    multi_time = sum(r["elapsed_sec"] for r in multiagent_results) / len(multiagent_results)
    lines.append(f"| avg_time_sec | {base_time:.1f} | {multi_time:.1f} | {multi_time-base_time:+.1f} | - |")
    
    lines.append("")
    
    # Per category comparison
    lines.append("## Par Catégorie")
    lines.append("")
    lines.append("| Catégorie | Système | Faithfulness | Answer Relevancy | Context Precision | Routing | N |")
    lines.append("|-----------|---------|--------------|------------------|-------------------|---------|---|")
    
    categories = set(r.get("categorie", "") for r in baseline_results) | set(r.get("categorie", "") for r in multiagent_results)
    categories.discard("")
    
    for cat in sorted(categories):
        for system_name, results in [("Baseline", baseline_results), ("Multi-Agents", multiagent_results)]:
            cat_results = [r for r in results if r.get("categorie", "") == cat]
            if not cat_results:
                continue
            
            faith = [r.get("faithfulness") for r in cat_results if r.get("faithfulness") is not None]
            rel = [r.get("answer_relevancy") for r in cat_results if r.get("answer_relevancy") is not None]
            prec = [r.get("context_precision") for r in cat_results if r.get("context_precision") is not None]
            routing = sum(1 for r in cat_results if r.get("routing_correct", False))
            
            f_avg = sum(faith)/len(faith) if faith else 0
            r_avg = sum(rel)/len(rel) if rel else 0
            p_avg = sum(prec)/len(prec) if prec else 0
            
            lines.append(f"| {cat} | {system_name} | {f_avg:.3f} | {r_avg:.3f} | {p_avg:.3f} | {routing}/{len(cat_results)} | {len(cat_results)} |")
    
    lines.append("")
    
    # Per question comparison
    lines.append("## Par Question")
    lines.append("")
    lines.append("| ID | Question | Catégorie | Baseline Faith. | Multi Faith. | Delta | Baseline Rel. | Multi Rel. | Delta |")
    lines.append("|----|----------|-----------|-----------------|--------------|-------|---------------|------------|-------|")
    
    for base in baseline_results:
        qid = base.get("id")
        multi = multiagent_by_id.get(qid)
        if not multi:
            continue
        
        bf = base.get("faithfulness") or 0
        mf = multi.get("faithfulness") or 0
        br = base.get("answer_relevancy") or 0
        mr = multi.get("answer_relevancy") or 0
        
        lines.append(f"| {qid} | {base.get('question', '')[:40]}... | {base.get('categorie', '')} | {bf:.3f} | {mf:.3f} | {mf-bf:+.3f} | {br:.3f} | {mr:.3f} | {mr-br:+.3f} |")
    
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Baseline Evaluation for AutoDrive")
    parser.add_argument("--max-questions", type=int, default=None, help="Max questions to evaluate")
    parser.add_argument("--skip-llm-eval", action="store_true", help="Skip LLM-based evaluation")
    args = parser.parse_args()
    
    logger.info("Starting baseline evaluation...")
    
    dataset = load_dataset(DATASET_FILE)
    logger.info(f"Loaded {len(dataset)} questions")
    
    existing = load_existing_results(BASELINE_RESULTS_FILE)
    logger.info(f"Found {len(existing)} existing baseline results")
    
    to_evaluate = [q for q in dataset if q["id"] not in existing]
    if args.max_questions:
        to_evaluate = to_evaluate[:args.max_questions]
    
    logger.info(f"Will evaluate {len(to_evaluate)} questions")
    
    if not to_evaluate:
        logger.info("All questions already evaluated.")
        baseline_results = list(existing.values())
    else:
        # Initialize baseline RAG
        if not args.skip_llm_eval:
            baseline = BaselineRAG()
        else:
            baseline = None
        
        baseline_results = list(existing.values())
        
        for i, question_data in enumerate(to_evaluate):
            logger.info(f"Progress: {i+1}/{len(to_evaluate)}")
            
            try:
                if args.skip_llm_eval:
                    session_id, history = get_or_create_session()
                    # Just process without eval
                    result = {
                        "id": question_data["id"],
                        "question": question_data["question"],
                        "categorie": question_data["categorie"],
                        "agent_attendu": question_data["agent_attendu"],
                        "agent_utilise": "baseline",
                        "ground_truth": question_data["ground_truth"],
                        "reponse": "",
                        "contexts": [],
                        "faithfulness": None,
                        "answer_relevancy": None,
                        "context_precision": None,
                        "routing_correct": False,
                        "off_topic_correct": False,
                        "elapsed_sec": 0,
                        "timestamp": datetime.now().isoformat(),
                    }
                else:
                    result = evaluate_single_question(question_data, baseline)
                
                baseline_results.append(result)
                save_result(BASELINE_RESULTS_FILE, result)
                
            except Exception as e:
                logger.error(f"Failed to evaluate {question_data['id']}: {e}")
                continue
    
    # Load multi-agent results for comparison
    multiagent_results = load_multiagent_results(Path("/home/tawil/Bureau/stage d'ete/resultats_eval.jsonl"))
    
    # Generate comparison
    global multiagent_by_id
    multiagent_by_id = {r["id"]: r for r in multiagent_results}
    
    comparison_md = generate_comparison(baseline_results, multiagent_results)
    
    with open(COMPARISON_FILE, "w", encoding="utf-8") as f:
        f.write(comparison_md)
    
    logger.info(f"Comparison saved to {COMPARISON_FILE}")
    
    # Print summary
    print("\n" + "="*80)
    print("BASELINE EVALUATION SUMMARY")
    print("="*80)
    
    if baseline_results:
        valid_faith = [r.get("faithfulness") for r in baseline_results if r.get("faithfulness") is not None]
        valid_rel = [r.get("answer_relevancy") for r in baseline_results if r.get("answer_relevancy") is not None]
        valid_prec = [r.get("context_precision") for r in baseline_results if r.get("context_precision") is not None]
        
        if valid_faith:
            print(f"Faithfulness: {sum(valid_faith)/len(valid_faith):.4f} (target: {TARGETS['faithfulness']})")
        if valid_rel:
            print(f"Answer Relevancy: {sum(valid_rel)/len(valid_rel):.4f} (target: {TARGETS['answer_relevancy']})")
        if valid_prec:
            print(f"Context Precision: {sum(valid_prec)/len(valid_prec):.4f} (target: {TARGETS['context_precision']})")
        
        routing_correct = sum(1 for r in baseline_results if r.get("routing_correct", False))
        print(f"Routing Accuracy: {routing_correct}/{len(baseline_results)} = {routing_correct/len(baseline_results)*100:.1f}%")
        
        avg_time = sum(r.get("elapsed_sec", 0) for r in baseline_results) / len(baseline_results)
        print(f"Avg response time: {avg_time:.1f}s")
    
    close_resources()
    logger.info("Baseline evaluation complete!")

if __name__ == "__main__":
    main()