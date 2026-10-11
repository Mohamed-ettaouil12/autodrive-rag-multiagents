import logging
import uuid
from typing import Dict, Any, List, Optional, Tuple

from graph.builder import build_graph
from memory import (
    create_session,
    save_message,
    load_history,
    close_driver,
    get_last_session_id,
    is_neo4j_available,
)

LOG_FORMAT = "%(asctime)s [CORE] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("core")

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def get_or_create_session(session_id: Optional[str] = None) -> Tuple[str, List[Dict[str, str]]]:
    """Get existing session or create new one. Returns (session_id, history)."""
    if session_id and session_id != "no-memory":
        # Verify session exists by trying to load history
        history = load_history(session_id)
        if history is not None:
            logger.info(f"Session rechargée: {session_id} - {len(history)} messages")
            return session_id, history

    # Create new session
    if is_neo4j_available():
        new_session_id = create_session()
        if new_session_id:
            logger.info(f"Nouvelle session créée: {new_session_id}")
            return new_session_id, []
        else:
            logger.warning("Neo4j non disponible - fonctionnement sans mémoire")
            return "no-memory", []
    else:
        logger.warning("Neo4j non disponible - fonctionnement sans mémoire")
        return "no-memory", []


def get_last_session() -> Tuple[Optional[str], List[Dict[str, str]]]:
    """Get the last session ID and its history."""
    if is_neo4j_available():
        session_id = get_last_session_id()
        if session_id:
            history = load_history(session_id)
            logger.info(f"Dernière session trouvée: {session_id} - {len(history)} messages")
            return session_id, history
    return None, []


def process_question(
    question: str,
    session_id: str,
    history: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Process a question through the LangGraph. Returns result with response, sources, agent_used."""
    graph = get_graph()

    initial_state = {
        "question": question,
        "question_reformulee": "",
        "agents": [],
        "documents": [],
        "reponse": "",
        "sources": [],
        "session_id": session_id,
        "historique": history,
    }

    result = graph.invoke(initial_state)

    response = result.get("reponse", "")
    sources = result.get("sources", [])
    agents = result.get("agents", [])
    agent_used = agents[0] if agents else "hors_sujet"

    return {
        "reponse": response,
        "sources": sources,
        "agent_utilise": agent_used,
        "session_id": session_id,
    }


def save_exchange(session_id: str, question: str, response: str) -> bool:
    """Save user question and assistant response to Neo4j."""
    if session_id == "no-memory":
        return False

    user_saved = save_message(session_id, "user", question)
    assistant_saved = save_message(session_id, "assistant", response)
    return user_saved and assistant_saved


def get_session_history(session_id: str, limit: int = 50) -> List[Dict[str, str]]:
    """Get full session history from Neo4j."""
    if session_id == "no-memory":
        return []
    return load_history(session_id, limit=limit)


def close_resources():
    """Close database connections."""
    close_driver()
    logger.info("Ressources fermées")