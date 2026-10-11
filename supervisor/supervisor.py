import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    ENABLE_STRUCTURED_OUTPUT,
    MAX_JSON_RETRIES,
    OLLAMA_JSON_FORMAT,
)
from schemas import RoutingDecision
from utils.structured_output import call_llm_structured, StructuredOutputError

PROMPT_PATH = Path(__file__).parent / "supervisor_prompt.txt"

LOG_FORMAT = "%(asctime)s [SUPERVISOR] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
)
logger = logging.getLogger("supervisor")

AVAILABLE_AGENTS = [
    "contrats",
    "assurances",
    "faq",
    "vehicules",
    "maintenance",
]

# Keyword-based routing as reliable fallback
AGENT_KEYWORDS = {
    "contrats": ["contrat", "tarif", "caution", "pnalit", "penalit", "document", "duree", "location", "kilometrage", "kilomtrage", "carburant", "age", "permis", "pieces", "conditions", "casa-", "rab-", "mrk-", "atlas-"],
    "assurances": ["assurance", "franchise", "garantie", "couverture", "sinistre", "responsabilit", "basic", "cdw", "premium", "full protection", "vol", "bris de glace"],
    "faq": ["horaire", "agence", "contact", "telephone", "email", "adresse", "ouvrir", "fermer", "jour ferie", "dimanche", "samedi", "categorie vehicule", "place", "conducteur", "entreprise", "touriste", "etranger", "automatique"],
    "vehicules": ["vehicule", "modele", "categorie", "dacia", "renault", "hyundai", "sandero", "clio", "i10", "suv", "berline", "monospace", "utilitaire", "transmission", "boite", "diesel", "essence", "climatisation", "gps", "disponible", "gamme"],
    "maintenance": ["vidange", "pneu", "frein", "batterie", "climatisation", "entretien", "maintenance", "seuil", "kilometrage", "km", "mois", "controle", "checkpoint", "inspection", "retour", "depart", "voyant", "urgence", "symptome", "etapes", "procedure"],
}

def _keyword_route(question: str) -> List[str]:
    """Route based on keyword matching - fast and reliable fallback."""
    q_lower = question.lower()
    scores = {}
    for agent, keywords in AGENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in q_lower)
        if score > 0:
            scores[agent] = score
    
    if not scores:
        return []
    
    # Return agent(s) with highest score
    max_score = max(scores.values())
    return [agent for agent, score in scores.items() if score == max_score]


def load_system_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _validate_agents(agents: List[str]) -> List[str]:
    valid = [a for a in agents if a in AVAILABLE_AGENTS]
    if len(valid) != len(agents):
        dropped = set(agents) - set(valid)
        logger.warning("Agents invalides ignorés : %s", dropped)
    return valid


def route(question: str) -> Dict[str, List[str]]:
    system_prompt = load_system_prompt()
    logger.info("Question reçue : %s", question)

    # Try structured output first
    if ENABLE_STRUCTURED_OUTPUT:
        try:
            decision = call_llm_structured(
                prompt=question,
                schema=RoutingDecision,
                system_prompt=system_prompt,
                model="mistral",
                temperature=0.0,
                max_retries=MAX_JSON_RETRIES,
                use_json_format=OLLAMA_JSON_FORMAT,
            )
            decision.agents = _validate_agents(decision.agents)
            if decision.agents:  # Only use if non-empty
                logger.info("Agents sélectionnés (structuré) : %s", decision.agents)
                return {"agents": decision.agents}
            else:
                logger.warning("Routing structuré retourné vide, fallback mots-clés")
        except StructuredOutputError as e:
            logger.warning("Échec routing structuré, fallback mots-clés : %s", e)

    # Fallback: keyword-based routing
    agents = _keyword_route(question)
    agents = _validate_agents(agents)
    logger.info("Agents sélectionnés (mots-clés) : %s", agents)
    return {"agents": agents}


if __name__ == "__main__":
    print("\n===============================")
    print(" AutoDrive Supervisor v1")
    print("===============================\n")

    while True:
        question = input("Question : ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        result = route(question)

        print("\n--- RÉSULTAT SUPERVISOR ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print()