import json
import logging
import re
import ollama
from typing import Dict, Any, List, Optional

from supervisor.supervisor import route
from hybrid_retrieval import retrieve_hybrid
from rag_autodrive import generate_answer
from memory import load_history, is_neo4j_available
from graph_query import (
    get_contrat,
    get_assurance_contrat,
    get_historique_vehicule,
    is_neo4j_available as graph_is_available,
)
from config import ENABLE_STRUCTURED_OUTPUT, MAX_JSON_RETRIES, OLLAMA_JSON_FORMAT, ENABLE_TABLE_TOOL
from schemas import ReformulatedQuestion
from pydantic import ValidationError
from tools.table_calculator import calculate_table, TableCalcResult
from utils.structured_output import call_llm_structured, StructuredOutputError

LOG_FORMAT = "%(asctime)s [NODES] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("nodes")

CONTRAT_REGEX = re.compile(r"[A-Z]{2,6}-\d{4}-\d{2,3}")
IMMAT_REGEX = re.compile(r"ADM-[A-F]-[A-Z]+-\d+-\d+")


REFORmulateur_PROMPT = """Tu es un assistant qui reformule une question en question autonome.

Contexte : conversation avec un utilisateur sur AutoDrive Maroc (location de véhicules).
Historique de la conversation :
{historique}

Question actuelle : {question}

RÈGLES :
- Réécris la question pour qu'elle soit compréhensible SANS l'historique
- Garde les références aux contrats, véhicules, clients mentionnés précédemment
- Si la question est déjà autonome, retourne-la telle quelle
- Réponds UNIQUEMENT au format JSON : {"question": "question reformulée"}

Question reformulée :"""


def superviseur(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state.get("question_reformulee") or state["question"]
    result = route(question)
    agents = result.get("agents", [])
    logger.info("Agents sélectionnés pour la question : %s", agents)
    return {"agents": agents}


def _run_agent(state: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
    """Fonction générique pour exécuter un agent RAG avec retrieval hybride."""
    question = state.get("question_reformulee") or state["question"]
    logger.info("Traitement par l'agent %s : %s", agent_name, question)

    documents, metadata = retrieve_hybrid(question, agent_name, top_k_final=5)
    reponse = generate_answer(question, documents)

    sources = []
    for m in metadata:
        source_file = m.get("source", "inconnu")
        sources.append(source_file)

    logger.info("Documents récupérés : %d", len(documents))
    logger.info("Sources : %s", sources)

    return {
        "documents": documents,
        "reponse": reponse,
        "sources": sources,
    }


def _enrich_with_graph(state: Dict[str, Any], agent_name: str) -> str:
    """Enrichit la question avec des données du graphe si numéro contrat ou immatriculation détecté."""
    question = state.get("question_reformulee") or state["question"]
    graph_context = ""
    
    if not graph_is_available():
        return graph_context
    
    contrat_match = CONTRAT_REGEX.search(question)
    immat_match = IMMAT_REGEX.search(question)
    
    if agent_name == "contrats" and contrat_match:
        numero = contrat_match.group(0)
        results = get_contrat(numero)
        if results:
            r = results[0]
            contrat = r.get("contrat", {})
            vehicules = r.get("vehicules", [])
            assurances = r.get("assurances", [])
            
            parts = [f"Contrat {numero}:"]
            for k, v in contrat.items():
                if k not in ("nom", "created_at", "updated_at"):
                    parts.append(f"  {k}: {v}")
            if vehicules:
                parts.append("  Véhicule(s) couvert(s):")
                for v in vehicules:
                    parts.append(f"    - {v.get('nom')}: {', '.join(f'{k}={v}' for k, v in v.items() if k not in ('nom', 'created_at', 'updated_at'))}")
            if assurances:
                parts.append("  Assurance(s) incluse(s):")
                for a in assurances:
                    parts.append(f"    - {a.get('nom')}: {', '.join(f'{k}={v}' for k, v in a.items() if k not in ('nom', 'created_at', 'updated_at'))}")
            graph_context = "\n".join(parts)
            logger.info("[GRAPH] Données graphe récupérées pour %s : %s", numero, graph_context[:200])
    
    elif agent_name == "assurances" and contrat_match:
        numero = contrat_match.group(0)
        results = get_assurance_contrat(numero)
        if results:
            parts = [f"Assurances du contrat {numero}:"]
            for r in results:
                assurance = r.get("assurance", {})
                garanties = r.get("garanties", [])
                exclusions = r.get("exclusions", [])
                parts.append(f"  Assurance: {assurance.get('nom')}")
                for k, v in assurance.items():
                    if k not in ("nom", "created_at", "updated_at"):
                        parts.append(f"    {k}: {v}")
                if garanties:
                    parts.append("    Garanties:")
                    for g in garanties:
                        parts.append(f"      - {g.get('nom')}: {', '.join(f'{k}={v}' for k, v in g.items() if k not in ('nom', 'created_at', 'updated_at'))}")
                if exclusions:
                    parts.append("    Exclusions:")
                    for e in exclusions:
                        parts.append(f"      - {e.get('nom')}: {', '.join(f'{k}={v}' for k, v in e.items() if k not in ('nom', 'created_at', 'updated_at'))}")
            graph_context = "\n".join(parts)
            logger.info("[GRAPH] Données graphe récupérées pour %s : %s", numero, graph_context[:200])
    
    elif (agent_name in ("contrats", "assurances", "vehicules", "maintenance")) and immat_match:
        immat = immat_match.group(0)
        results = get_historique_vehicule(immat)
        if results:
            r = results[0]
            vehicule = r.get("vehicule", {})
            maintenances = r.get("maintenances", [])
            parts = [f"Véhicule {immat}:"]
            for k, v in vehicule.items():
                if k not in ("nom", "created_at", "updated_at"):
                    parts.append(f"  {k}: {v}")
            if maintenances:
                parts.append("  Historique maintenance:")
                for m in maintenances:
                    parts.append(f"    - {m.get('nom')}: {', '.join(f'{k}={v}' for k, v in m.items() if k not in ('nom', 'created_at', 'updated_at'))}")
            graph_context = "\n".join(parts)
            logger.info("[GRAPH] Données graphe récupérées pour véhicule %s : %s", immat, graph_context[:200])
    
    return graph_context


def _is_numerical_question(question: str) -> bool:
    """Check if question asks for a specific numerical value."""
    numerical_keywords = [
        "combien", "quel est le", "quelle est la", "à combien", "a combien",
        "tarif", "prix", "coût", "cout", "caution", "dépôt", "depot",
        "franchise", "pénalité", "penalite", "penalité", "pourcentage",
        "kilométrage", "kilometrage", "km", "seuil", "fréquence", "frequence",
        "frais", "montant", "tarif journalier", "tarif mensuel",
        "supplémentaire", "supplementaire", "excédent", "excedent",
        "vidange", "pneu", "frein", "batterie", "climatisation",
        "taux", "taux journalier", "coût journalier"
    ]
    q_lower = question.lower()
    return any(kw in q_lower for kw in numerical_keywords)


def _enrich_with_table_tool(question: str, category: str) -> Optional[str]:
    """Call table calculator and format result for context injection."""
    if not ENABLE_TABLE_TOOL:
        return None
    
    if not _is_numerical_question(question):
        return None
    
    # Map agent category to calculator category
    cat_map = {
        "contrats": "contrats",
        "assurances": "assurances",
        "maintenance": "maintenance",
    }
    calc_category = cat_map.get(category, category)
    
    try:
        result = calculate_table(question, calc_category)
        if result:
            logger.info("[TABLE_TOOL] %s -> %s %s (formula: %s)", 
                       question, result.value, result.unit, result.formula)
            return (
                f"RÉSULTAT CALCUL DÉTERMINISTE (outil table_calculator) :\n"
                f"Valeur : {result.value} {result.unit}\n"
                f"Formule : {result.formula}\n"
                f"Sources : {', '.join(result.source_chunks) if result.source_chunks else 'N/A'}\n"
                f"---\n"
            )
    except Exception as e:
        logger.warning("[TABLE_TOOL] Erreur pour '%s' : %s", question, e)
    
    return None


def agent_contrats(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state.get("question_reformulee") or state["question"]
    logger.info("Traitement par l'agent contrats : %s", question)
    
    graph_context = _enrich_with_graph(state, "contrats")
    table_context = _enrich_with_table_tool(question, "contrats")
    
    documents, metadata = retrieve_hybrid(question, "contrats", top_k_final=5)
    
    # Inject contexts: table tool first (deterministic), then graph, then retrieved docs
    context_parts = []
    if table_context:
        context_parts.append(table_context)
    if graph_context:
        context_parts.append(f"Données du graphe :\n{graph_context}")
    context_parts.extend(documents)
    documents = context_parts
    
    reponse = generate_answer(question, documents)
    
    sources = []
    for m in metadata:
        source_file = m.get("source", "inconnu")
        sources.append(source_file)
    
    logger.info("Documents récupérés : %d", len(documents))
    logger.info("Sources : %s", sources)
    
    return {
        "documents": documents,
        "reponse": reponse,
        "sources": sources,
    }


def agent_assurances(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state.get("question_reformulee") or state["question"]
    logger.info("Traitement par l'agent assurances : %s", question)
    
    graph_context = _enrich_with_graph(state, "assurances")
    table_context = _enrich_with_table_tool(question, "assurances")
    
    documents, metadata = retrieve_hybrid(question, "assurances", top_k_final=5)
    
    context_parts = []
    if table_context:
        context_parts.append(table_context)
    if graph_context:
        context_parts.append(f"Données du graphe :\n{graph_context}")
    context_parts.extend(documents)
    documents = context_parts
    
    reponse = generate_answer(question, documents)
    
    sources = []
    for m in metadata:
        source_file = m.get("source", "inconnu")
        sources.append(source_file)
    
    logger.info("Documents récupérés : %d", len(documents))
    logger.info("Sources : %s", sources)
    
    return {
        "documents": documents,
        "reponse": reponse,
        "sources": sources,
    }


def agent_faq(state: Dict[str, Any]) -> Dict[str, Any]:
    return _run_agent(state, "faq")


def agent_vehicules(state: Dict[str, Any]) -> Dict[str, Any]:
    return _run_agent(state, "vehicules")


def agent_maintenance(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state.get("question_reformulee") or state["question"]
    logger.info("Traitement par l'agent maintenance : %s", question)
    
    table_context = _enrich_with_table_tool(question, "maintenance")
    
    documents, metadata = retrieve_hybrid(question, "maintenance", top_k_final=5)
    
    context_parts = []
    if table_context:
        context_parts.append(table_context)
    context_parts.extend(documents)
    documents = context_parts
    
    reponse = generate_answer(question, documents)
    
    sources = []
    for m in metadata:
        source_file = m.get("source", "inconnu")
        sources.append(source_file)
    
    logger.info("Documents récupérés : %d", len(documents))
    logger.info("Sources : %s", sources)
    
    return {
        "documents": documents,
        "reponse": reponse,
        "sources": sources,
    }


def hors_sujet(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state.get("question_reformulee") or state["question"]
    logger.info("Question hors sujet : %s", question)

    reponse = (
        "Je ne peux pas répondre à cette question car elle est hors de mon périmètre d'expertise "
        "(contrats, assurances, FAQ, véhicules, maintenance). "
        "N'hésitez pas à me poser une question sur l'un de ces sujets."
    )

    return {
        "reponse": reponse,
        "documents": [],
        "sources": [],
    }


def reformulateur(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state["question"]
    session_id = state.get("session_id", "")
    historique = state.get("historique", [])

    logger.info("Reformulateur: question='%s', session_id='%s', historique=%d messages",
                question, session_id, len(historique))

    if not historique:
        logger.info("Pas d'historique, pas de reformulation nécessaire")
        return {"question_reformulee": question}

    if not is_neo4j_available():
        logger.warning("Neo4j indisponible, pas de reformulation")
        return {"question_reformulee": question}

    # Construire l'historique formaté
    hist_lines = []
    for msg in historique:
        role = "Utilisateur" if msg["role"] == "user" else "Assistant"
        hist_lines.append(f"{role}: {msg['content']}")
    historique_str = "\n".join(hist_lines)

    prompt = REFORmulateur_PROMPT.format(historique=historique_str, question=question)

    try:
        reformulated_obj = call_llm_structured(
            prompt=prompt,
            schema=ReformulatedQuestion,
            model="mistral",
            temperature=0.0,
            max_retries=MAX_JSON_RETRIES,
            use_json_format=OLLAMA_JSON_FORMAT,
        )
        reformulated = reformulated_obj.question
        logger.info("Question reformulée: '%s' -> '%s'", question, reformulated)
        return {"question_reformulee": reformulated}
    except StructuredOutputError as e:
        logger.error("Échec reformulation structurée: %s. Utilisation question originale.", e)
        return {"question_reformulee": question}