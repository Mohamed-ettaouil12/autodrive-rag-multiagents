import logging
from typing import List, Dict, Any, Optional

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

LOG_FORMAT = "%(asctime)s [GRAPH] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("graph_query")

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
            logger.warning("Neo4j injoignable: %s. Fonctionnement sans graphe.", e)
            _driver = None
    return _driver


def is_neo4j_available() -> bool:
    _get_driver()
    return _neo4j_available


def _run_query(query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    driver = _get_driver()
    if not driver:
        logger.warning("Neo4j indisponible, retour liste vide")
        return []
    
    try:
        with driver.session() as session:
            result = session.run(query, params)
            return [dict(record) for record in result]
    except Exception as e:
        logger.warning("Erreur exécution requête Cypher: %s", e)
        return []


def get_contrat(numero: str) -> List[Dict[str, Any]]:
    """Retourne le contrat + véhicule couvert + assurance incluse (noms + propriétés clés)."""
    query = """
    MATCH (c:Contrat {nom: $numero})
    OPTIONAL MATCH (c)-[:COUVRE]->(v:Vehicule)
    OPTIONAL MATCH (c)-[:INCLUT]->(a:Assurance)
    RETURN c, collect(DISTINCT v) as vehicules, collect(DISTINCT a) as assurances
    """
    results = _run_query(query, {"numero": numero})
    if not results:
        logger.info("Données graphe récupérées pour %s : aucun résultat", numero)
        return []
    
    record = results[0]
    contrat = dict(record["c"])
    vehicules = [dict(v) for v in record["vehicules"] if v]
    assurances = [dict(a) for a in record["assurances"] if a]
    
    logger.info("Données graphe récupérées pour %s : contrat=%s, vehicules=%d, assurances=%d",
                numero, contrat.get("nom"), len(vehicules), len(assurances))
    return [{
        "contrat": contrat,
        "vehicules": vehicules,
        "assurances": assurances
    }]


def get_assurance_contrat(numero_contrat: str) -> List[Dict[str, Any]]:
    """Retourne les garanties et franchises d'un contrat via ses assurances."""
    query = """
    MATCH (c:Contrat {nom: $numero})-[:INCLUT]->(a:Assurance)
    OPTIONAL MATCH (a)-[:INCLUT]->(g)
    OPTIONAL MATCH (a)-[:EXCLUT]->(e)
    RETURN a, collect(DISTINCT g) as garanties, collect(DISTINCT e) as exclusions
    """
    results = _run_query(query, {"numero": numero_contrat})
    if not results:
        logger.info("Données graphe récupérées pour %s : aucune assurance trouvée", numero_contrat)
        return []
    
    output = []
    for record in results:
        assurance = dict(record["a"])
        garanties = [dict(g) for g in record["garanties"] if g]
        exclusions = [dict(e) for e in record["exclusions"] if e]
        output.append({
            "assurance": assurance,
            "garanties": garanties,
            "exclusions": exclusions
        })
    
    logger.info("Données graphe récupérées pour %s : %d assurance(s)", numero_contrat, len(output))
    return output


def get_historique_vehicule(immatriculation: str) -> List[Dict[str, Any]]:
    """Retourne les maintenances liées à un véhicule."""
    query = """
    MATCH (v:Vehicule {nom: $immatriculation})-[:A_HISTORIQUE]->(m:Maintenance)
    RETURN v, collect(m) as maintenances
    """
    results = _run_query(query, {"immatriculation": immatriculation})
    if not results:
        logger.info("Données graphe récupérées pour véhicule %s : aucune maintenance", immatriculation)
        return []
    
    record = results[0]
    vehicule = dict(record["v"])
    maintenances = [dict(m) for m in record["maintenances"] if m]
    
    logger.info("Données graphe récupérées pour véhicule %s : %d maintenance(s)", 
                immatriculation, len(maintenances))
    return [{
        "vehicule": vehicule,
        "maintenances": maintenances
    }]


def recherche_entite(nom_partiel: str) -> List[Dict[str, Any]]:
    """Recherche floue CONTAINS insensible à la casse sur tous types d'entités."""
    query = """
    MATCH (n)
    WHERE toLower(n.nom) CONTAINS toLower($nom)
    RETURN labels(n)[0] as type, n
    LIMIT 50
    """
    results = _run_query(query, {"nom": nom_partiel})
    output = []
    for record in results:
        node = dict(record["n"])
        node["_type"] = record["type"]
        output.append(node)
    
    logger.info("Recherche entité '%s' : %d résultat(s)", nom_partiel, len(output))
    return output


if __name__ == "__main__":
    # Tests rapides
    print("Test get_contrat:")
    print(get_contrat("ATLAS-2024-007"))
    print("\nTest get_assurance_contrat:")
    print(get_assurance_contrat("ATLAS-2024-007"))
    print("\nTest get_historique_vehicule:")
    print(get_historique_vehicule("ADM-A-CASABLANCA-01-001"))
    print("\nTest recherche_entite:")
    print(recherche_entite("CASA"))