#!/usr/bin/env python3
"""
ReAct Agent Node - Fixed version.
The agent receives routing ONCE from supervisor and restricts tools to the targeted agent.
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional

import ollama
from pydantic import ValidationError

from config import (
    ENABLE_REACT,
    MAX_REACT_STEPS,
    REACT_TOKEN_BUDGET,
    REACT_TIME_BUDGET_SEC,
    ENABLE_TABLE_TOOL,
    ENABLE_STRUCTURED_OUTPUT,
    OLLAMA_JSON_FORMAT,
    MAX_JSON_RETRIES,
)
from schemas import (
    ToolCall,
    ToolResult,
    ReActStep,
    ReActState,
    RoutingDecision,
    ReformulatedQuestion,
)
from utils.structured_output import call_llm_structured, StructuredOutputError
from hybrid_retrieval import retrieve_hybrid
from rag_autodrive import generate_answer
from graph_query import is_neo4j_available as graph_is_available
from tools.table_calculator import calculate_table, TableCalcResult
from tools.text2cypher import text2cypher, Text2CypherInput
from tools.clarification import ask_clarification, ClarificationInput
from tools.final_answer import final_answer, FinalAnswerInput

LOG_FORMAT = "%(asctime)s [REACT] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("react_agent")


REACT_SYSTEM_PROMPT = """Tu es un agent ReAct pour AutoDrive Maroc. Réponds en choisissant UN outil par étape.

OUTILS DISPONIBLES (selon agent ciblé) :
1. search_hybrid(question: str) -> Recherche documentaire
2. calculate_table(question: str) -> Calcul DÉTERMINISTE (tarifs, franchises, cautions, seuils maintenance). JAMAIS calculer toi-même.
3. text2cypher(query_type: str, identifier: str) -> Données graphe Neo4j (si contrat/immat détecté)
4. final_answer(answer: str, sources: list[str], confidence: float) -> TERMINE

RÈGLES STRICTES (OBLIGATOIRES) :
- Max 3 étapes. Si info suffisante -> final_answer.
- QUESTION CHIFFRÉE (tarif, franchise, caution, km, seuil, vidange, pneu, frein, batterie, climatisation, pourcentage, montant) -> OBLIGATOIRE calculate_table EN PREMIER.
- Neo4j UNIQUEMENT via text2cypher (si contrat/immat détecté dans question).
- INTERDICTION de répondre une valeur chiffrée sans résultat d'outil calculate_table.
- Si calculate_table n'a pas encore répondu, l'étape suivante OBLIGATOIRE est calculate_table.
- Si ambigu -> final_answer confidence<0.5 + "Il faut préciser..."
- Si sans réponse -> final_answer confidence=0 + "Information non trouvée"

JSON: {"tool": "nom", "args": {...}}"""


# Tool implementations
def tool_search_hybrid(question: str, category: str) -> ToolResult:
    try:
        docs, metadatas = retrieve_hybrid(question, category, top_k_final=5)
        sources = [m.get("source", "inconnu") for m in metadatas]
        return ToolResult(
            tool="search_hybrid",
            success=True,
            result={"documents": docs, "metadatas": metadatas, "sources": sources},
        )
    except Exception as e:
        return ToolResult(tool="search_hybrid", success=False, error=str(e))


def tool_calculate_table(question: str, category: str) -> ToolResult:
    if not ENABLE_TABLE_TOOL:
        return ToolResult(tool="calculate_table", success=False, error="Outil désactivé")
    try:
        result = calculate_table(question, category)
        if result:
            return ToolResult(
                tool="calculate_table",
                success=True,
                result={"value": result.value, "unit": result.unit, "formula": result.formula, "source_chunks": result.source_chunks},
            )
        return ToolResult(tool="calculate_table", success=True, result=None, error="Aucune valeur trouvée")
    except Exception as e:
        return ToolResult(tool="calculate_table", success=False, error=str(e))


def tool_text2cypher(query_type: str, identifier: str) -> ToolResult:
    try:
        input_data = Text2CypherInput(query_type=query_type, identifier=identifier)
        result = text2cypher(input_data)
        return ToolResult(tool="text2cypher", success=result.success, result=result.data, error=result.error)
    except Exception as e:
        return ToolResult(tool="text2cypher", success=False, error=str(e))


def tool_final_answer(answer: str, sources: List[str], confidence: float) -> ToolResult:
    try:
        input_data = FinalAnswerInput(answer=answer, sources=sources, confidence=confidence)
        result = final_answer(input_data)
        return ToolResult(tool="final_answer", success=True, result=result.model_dump())
    except Exception as e:
        return ToolResult(tool="final_answer", success=False, error=str(e))


TOOL_MAP = {
    "search_hybrid": tool_search_hybrid,
    "calculate_table": tool_calculate_table,
    "text2cypher": tool_text2cypher,
    "final_answer": tool_final_answer,
}

# Category mapping for calculate_table
CALC_CATEGORY_MAP = {
    "contrats": "contrats",
    "assurances": "assurances",
    "maintenance": "maintenance",
    "vehicules": "contrats",
    "faq": "contrats",
}

# Agent -> allowed tools mapping
AGENT_TOOLS = {
    "contrats": ["search_hybrid", "calculate_table", "text2cypher", "final_answer"],
    "assurances": ["search_hybrid", "calculate_table", "text2cypher", "final_answer"],
    "vehicules": ["search_hybrid", "calculate_table", "text2cypher", "final_answer"],
    "maintenance": ["search_hybrid", "calculate_table", "final_answer"],
    "faq": ["search_hybrid", "final_answer"],
}

# Agent -> search category mapping
AGENT_CATEGORY = {
    "contrats": "contrats",
    "assurances": "assurances",
    "vehicules": "vehicules",
    "maintenance": "maintenance",
    "faq": "faq",
}

# Numerical keywords that require calculate_table
NUMERICAL_KEYWORDS = [
    "combien", "quel est le", "quelle est la", "à combien", "a combien",
    "tarif", "prix", "coût", "cout", "caution", "dépôt", "depot",
    "franchise", "pénalité", "penalite", "penalité", "pourcentage",
    "kilométrage", "kilometrage", "km", "seuil", "fréquence", "frequence",
    "frais", "montant", "tarif journalier", "tarif mensuel",
    "supplémentaire", "supplementaire", "excédent", "excedent",
    "vidange", "pneu", "frein", "batterie", "climatisation",
    "taux", "taux journalier", "coût journalier"
]


def is_numerical_question(question: str) -> bool:
    q_lower = question.lower()
    return any(kw in q_lower for kw in NUMERICAL_KEYWORDS)


def has_contract_or_immat(question: str) -> bool:
    import re
    contrat_match = re.search(r"[A-Z]{2,6}-\d{4}-\d{2,3}", question)
    immat_match = re.search(r"ADM-[A-F]-[A-Z]+-\d+-\d+", question)
    return bool(contrat_match or immat_match)


def build_react_prompt(question: str, history: List[ReActStep], context: str, agent_cible: str, last_tool_result: Any = None) -> str:
    steps_str = ""
    for step in history:
        result_str = json.dumps(step.result, ensure_ascii=False)[:300]
        steps_str += f"\nÉtape {step.step}: {step.tool}({json.dumps(step.args, ensure_ascii=False)}) -> {result_str}"

    # Build allowed tools list for this agent
    allowed = AGENT_TOOLS.get(agent_cible, AGENT_TOOLS["faq"])
    allowed_str = ", ".join(allowed)

    # Inject last tool result if available (VERIFIED DATA instruction)
    tool_result_injection = ""
    if last_tool_result:
        tool_result_injection = f"""

⚠️ RÉSULTAT D'OUTIL VÉRIFIÉ (utilise-le TEL QUEL, ne recalcule pas, ne l'ignore pas) :
{json.dumps(last_tool_result, ensure_ascii=False, indent=2)}"""

    return f"""Question: {question}
Agent ciblé: {agent_cible} (outils: {allowed_str})

Contexte:
{context if context else "(aucun)"}

Historique:
{steps_str if steps_str else "(début)"}{tool_result_injection}

Choisis le PROCHAIN outil. JSON uniquement."""


def agent_react(state: Dict[str, Any]) -> Dict[str, Any]:
    if not ENABLE_REACT:
        logger.warning("ENABLE_REACT=False but agent_react called")
        return {"reponse": "Erreur: ReAct désactivé", "sources": [], "documents": []}

    question = state.get("question_reformulee") or state["question"]
    session_id = state.get("session_id", "")

    logger.info("ReAct Agent: '%s'", question)

    # Get routing from supervisor (already done upstream)
    agents = state.get("agents", [])
    if not agents:
        logger.warning("No agents from supervisor, defaulting to faq")
        agents = ["faq"]

    agent_cible = agents[0] if agents else "faq"
    search_category = AGENT_CATEGORY.get(agent_cible, "faq")
    allowed_tools = AGENT_TOOLS.get(agent_cible, AGENT_TOOLS["faq"])

    logger.info("Routing reçu du superviseur: %s (catégorie recherche: %s, outils: %s)", agent_cible, search_category, allowed_tools)

    react_state = ReActState(start_time=time.time(), agent_cible=agent_cible)
    context_parts = []
    all_sources = []

    # Initial retrieval for the targeted agent ONLY
    try:
        step_start = time.time()
        docs, metadatas = retrieve_hybrid(question, search_category, top_k_final=5)
        step_time = time.time() - step_start
        logger.info("Initial retrieval: %d docs in %.2fs", len(docs), step_time)
        context_parts.extend(docs)
        all_sources.extend([m.get("source", "inconnu") for m in metadatas])
    except Exception as e:
        logger.warning("Initial retrieval failed: %s", e)

    # Check for contract/immat patterns for potential graph queries
    contrat_match = None
    immat_match = None
    import re
    contrat_match = re.search(r"[A-Z]{2,6}-\d{4}-\d{2,3}", question)
    immat_match = re.search(r"ADM-[A-F]-[A-Z]+-\d+-\d+", question)

    # ReAct Loop - NO RE-ROUTING
    for step_num in range(1, MAX_REACT_STEPS + 1):
        step_start = time.time()
        elapsed = time.time() - react_state.start_time
        if elapsed > REACT_TIME_BUDGET_SEC or react_state.total_tokens > REACT_TOKEN_BUDGET:
            logger.warning("Budget exceeded: time=%.1fs, tokens=%d", elapsed, react_state.total_tokens)
            break

        context_str = "\n\n---\n\n".join(context_parts[:8]) if context_parts else "Aucun contexte"
        prompt = build_react_prompt(question, react_state.steps, context_str, agent_cible, react_state.last_tool_result)

        try:
            tool_call = call_llm_structured(
                prompt=prompt,
                schema=ToolCall,
                system_prompt=REACT_SYSTEM_PROMPT,
                model="mistral", temperature=0.0, max_retries=MAX_JSON_RETRIES, use_json_format=OLLAMA_JSON_FORMAT,
            )
        except StructuredOutputError as e:
            logger.error("Tool call parse failed: %s", e)
            break

        tool_name = tool_call.tool
        tool_args = tool_call.args

        # FORCE calculate_table for numerical questions if not yet called
        if is_numerical_question(question):
            calc_called = any(s.tool == "calculate_table" for s in react_state.steps)
            if not calc_called and tool_name != "calculate_table" and "calculate_table" in allowed_tools:
                logger.info("Question chiffrée détectée, forçage calculate_table au lieu de %s", tool_name)
                tool_name = "calculate_table"
                tool_args = {"question": question}

        # Validate tool is allowed for this agent
        if tool_name not in allowed_tools:
            logger.warning("Outil '%s' non autorisé pour agent '%s', ignoré", tool_name, agent_cible)
            tool_result = ToolResult(tool=tool_name, success=False, error=f"Outil non autorisé pour {agent_cible}")
            step = ReActStep(step=step_num, tool=tool_name, args=tool_args, result=tool_result.model_dump())
            react_state.steps.append(step)
            continue

        logger.info("Étape %d: %s(%s)", step_num, tool_name, tool_args)

        # Fix category for calculate_table
        if tool_name == "calculate_table":
            tool_args["category"] = CALC_CATEGORY_MAP.get(search_category, "contrats")
        elif tool_name == "search_hybrid":
            tool_args["category"] = search_category

        tool_func = TOOL_MAP.get(tool_name)
        if not tool_func:
            tool_result = ToolResult(tool=tool_name, success=False, error=f"Outil inconnu: {tool_name}")
        else:
            tool_result = tool_func(**tool_args)

        step = ReActStep(step=step_num, tool=tool_name, args=tool_args, result=tool_result.model_dump() if tool_result else {})
        react_state.steps.append(step)
        step_time = time.time() - step_start
        logger.info("Étape %d terminée en %.2fs: %s", step_num, step_time, "success" if tool_result.success else "failed")

        # Store last tool result for injection into next prompt
        if tool_result.success and tool_result.result:
            react_state.last_tool_result = tool_result.result

        # Handle result
        if tool_result.success and tool_result.result:
            if tool_name == "search_hybrid":
                data = tool_result.result
                context_parts.extend(data.get("documents", []))
                all_sources.extend(data.get("sources", []))
            elif tool_name == "calculate_table":
                data = tool_result.result
                if data:
                    context_parts.append(f"CALCUL VÉRIFIÉ: {data['value']} {data['unit']} ({data['formula']})")
            elif tool_name == "text2cypher":
                data = tool_result.result
                if data:
                    context_parts.append(f"GRAPHE: {json.dumps(data, ensure_ascii=False)}")
            elif tool_name == "final_answer":
                data = tool_result.result
                react_state.finished = True
                react_state.final_answer = data.get("answer", "")
                react_state.abstention_reason = data.get("abstention_reason")
                break

        # Auto-call final_answer if we have context and this is last step or calculate_table succeeded
        if step_num == MAX_REACT_STEPS or (tool_name == "calculate_table" and tool_result.success and tool_result.result):
            if not react_state.finished:
                try:
                    # Build final answer with VERIFIED DATA instruction
                    context_for_answer = "\n\n---\n\n".join(context_parts[:8])
                    if react_state.last_tool_result:
                        context_for_answer += f"\n\n⚠️ DONNÉE VÉRIFIÉE (outil) : {json.dumps(react_state.last_tool_result, ensure_ascii=False)}"
                    answer = generate_answer(question, [context_for_answer])
                    final_tool = tool_final_answer(answer, list(dict.fromkeys(all_sources)), 0.85)
                    if final_tool.success:
                        react_state.finished = True
                        react_state.final_answer = final_tool.result.get("answer", answer)
                except Exception as e:
                    logger.error("Auto final_answer failed: %s", e)
                break

    # Fallback generation
    if not react_state.finished:
        try:
            context_for_answer = "\n\n---\n\n".join(context_parts[:8])
            if react_state.last_tool_result:
                context_for_answer += f"\n\n⚠️ DONNÉE VÉRIFIÉE (outil) : {json.dumps(react_state.last_tool_result, ensure_ascii=False)}"
            answer = generate_answer(question, [context_for_answer])
            react_state.final_answer = answer
            react_state.abstention_reason = "Limite étapes atteinte"
        except Exception as e:
            react_state.final_answer = "Erreur génération réponse."
            react_state.abstention_reason = "Erreur"

    unique_sources = list(dict.fromkeys(all_sources))
    total_time = time.time() - react_state.start_time
    logger.info("ReAct done: %d steps, %.2fs total, answer: %s...", len(react_state.steps), total_time, react_state.final_answer[:80])

    return {
        "reponse": react_state.final_answer,
        "sources": unique_sources,
        "documents": context_parts,
        "react_steps": [s.model_dump() for s in react_state.steps],
        "react_abstention_reason": react_state.abstention_reason,
        "agent_utilise": agent_cible,
    }


if __name__ == "__main__":
    test_state = {"question": "Quel est le tarif journalier du contrat CASA-2024-014 ?", "session_id": "test", "agents": ["contrats"]}
    result = agent_react(test_state)
    print(json.dumps(result, indent=2, ensure_ascii=False))