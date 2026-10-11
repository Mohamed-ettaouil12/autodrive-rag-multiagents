import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

LOG_FORMAT = "%(asctime)s [MEMORY] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("memory")

_driver = None
_neo4j_available = None


def _get_driver():
    global _driver, _neo4j_available
    if _driver is None:
        try:
            _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            _driver.verify_connectivity()
            _neo4j_available = True
            logger.info("Connexion Neo4j établie: %s", NEO4J_URI)
        except (ServiceUnavailable, AuthError) as e:
            _neo4j_available = False
            logger.error("Neo4j injoignable: %s. Fonctionnement sans mémoire.", e)
            _driver = None
    return _driver


def is_neo4j_available() -> bool:
    _get_driver()
    return _neo4j_available


def create_session() -> Optional[str]:
    """Crée une nouvelle session et retourne son UUID."""
    driver = _get_driver()
    if not driver:
        return None

    session_id = str(uuid.uuid4())
    date_debut = datetime.now().isoformat()

    query = """
    CREATE (s:Session {id: $id, date_debut: $date_debut})
    RETURN s.id as id
    """

    try:
        with driver.session() as session:
            result = session.run(query, id=session_id, date_debut=date_debut)
            record = result.single()
            if record:
                logger.info("Session créée: %s", session_id)
                return record["id"]
    except Exception as e:
        logger.error("Erreur création session: %s", e)
    return None


def save_message(session_id: str, role: str, content: str) -> bool:
    """Sauvegarde un message dans la session."""
    driver = _get_driver()
    if not driver:
        return False

    timestamp = datetime.now().isoformat()

    query = """
    MATCH (s:Session {id: $session_id})
    CREATE (m:Message {role: $role, content: $content, timestamp: $timestamp, ordre: $ordre})
    CREATE (s)-[:CONTIENT]->(m)
    RETURN m
    """

    try:
        with driver.session() as session:
            # Get current message count for ordering
            count_query = """
            MATCH (s:Session {id: $session_id})-[:CONTIENT]->(m:Message)
            RETURN count(m) as count
            """
            count_result = session.run(count_query, session_id=session_id)
            count_record = count_result.single()
            ordre = (count_record["count"] if count_record else 0) + 1

            session.run(query, session_id=session_id, role=role, content=content, timestamp=timestamp, ordre=ordre)
            logger.debug("Message sauvegardé: session=%s, role=%s, ordre=%d", session_id, role, ordre)
            return True
    except Exception as e:
        logger.error("Erreur sauvegarde message: %s", e)
        return False


def load_history(session_id: str, limit: int = 10) -> List[Dict[str, str]]:
    """Charge l'historique des 10 derniers messages."""
    driver = _get_driver()
    if not driver:
        return []

    query = """
    MATCH (s:Session {id: $session_id})-[:CONTIENT]->(m:Message)
    RETURN m.role as role, m.content as content
    ORDER BY m.ordre DESC
    LIMIT $limit
    """

    try:
        with driver.session() as session:
            result = session.run(query, session_id=session_id, limit=limit)
            messages = []
            for record in result:
                messages.append({"role": record["role"], "content": record["content"]})
            # Reverse to get chronological order
            messages.reverse()
            logger.info("Historique chargé: %d messages pour session %s", len(messages), session_id)
            return messages
    except Exception as e:
        logger.error("Erreur chargement historique: %s", e)
        return []


def get_last_session_id() -> Optional[str]:
    """Récupère l'ID de la session la plus récente."""
    driver = _get_driver()
    if not driver:
        return None

    query = """
    MATCH (s:Session)
    RETURN s.id as id
    ORDER BY s.date_debut DESC
    LIMIT 1
    """

    try:
        with driver.session() as session:
            result = session.run(query)
            record = result.single()
            if record:
                logger.info("Dernière session trouvée: %s", record["id"])
                return record["id"]
    except Exception as e:
        logger.error("Erreur récupération dernière session: %s", e)
    return None


def save_feedback(session_id: str, question: str, reponse: str, utile: bool) -> bool:
    """Sauvegarde un feedback utilisateur sur une réponse."""
    driver = _get_driver()
    if not driver:
        return False

    timestamp = datetime.now().isoformat()
    feedback_id = str(uuid.uuid4())

    query = """
    MATCH (s:Session {id: $session_id})
    CREATE (f:Feedback {
        id: $feedback_id,
        question: $question,
        reponse: $reponse,
        utile: $utile,
        timestamp: $timestamp
    })
    CREATE (s)-[:A_FEEDBACK]->(f)
    RETURN f
    """

    try:
        with driver.session() as session:
            session.run(
                query,
                session_id=session_id,
                feedback_id=feedback_id,
                question=question,
                reponse=reponse,
                utile=utile,
                timestamp=timestamp,
            )
            logger.info("Feedback sauvegardé: session=%s, utile=%s", session_id, utile)
            return True
    except Exception as e:
        logger.error("Erreur sauvegarde feedback: %s", e)
        return False


def close_driver():
    global _driver
    if _driver:
        _driver.close()
        _driver = None
        logger.info("Connexion Neo4j fermée")