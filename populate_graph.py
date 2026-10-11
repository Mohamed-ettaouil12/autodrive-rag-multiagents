import json
import logging
import glob
from pathlib import Path
from typing import Dict, List, Set, Tuple

from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

LOG_FORMAT = "%(asctime)s [GRAPH] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("populate_graph")

INVALID_NAMES = {"", "non_spécifié", "non_specifie", "inconnu", "unknown", "n/a", "null", "none", "non_renseigné"}

VALID_RELATION_TYPES = {
    "COUVRE",
    "INCLUT",
    "APPARTIENT_A",
    "A_HISTORIQUE",
    "EXCLUT",
    "A_SIGNE",
    "APPLIQUE_A",
}

NODE_LABELS = {
    "Contrat",
    "Vehicule",
    "Assurance",
    "Client",
    "Politique",
    "Categorie",
    "Maintenance",
    "Sinistre",
}


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def is_valid_name(name: str) -> bool:
    if not name:
        return False
    return name.strip().lower() not in INVALID_NAMES


def clean_entity_name(name) -> str:
    if isinstance(name, dict):
        # Prendre la première valeur string trouvée dans le dict
        for v in name.values():
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    if isinstance(name, list):
        name = name[0] if name else ""
    return name.strip() if name else ""


def load_all_extractions(pattern: str = "extraction_*.jsonl") -> List[dict]:
    files = glob.glob(pattern)
    if not files:
        logger.error("Aucun fichier d'extraction trouvé: %s", pattern)
        return []
    
    all_records = []
    for f in files:
        logger.info("Lecture: %s", f)
        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning("Ligne invalide dans %s: %s", f, e)
    logger.info("Total enregistrements: %d", len(all_records))
    return all_records


def filter_entities_and_relations(entities: List[dict], relations: List[dict]) -> Tuple[List[dict], List[dict], Dict[str, int]]:
    stats = {
        "entities_total": len(entities),
        "entities_kept": 0,
        "entities_rejected_empty": 0,
        "entities_rejected_invalid": 0,
        "relations_total": len(relations),
        "relations_kept": 0,
        "relations_rejected_invalid_endpoints": 0,
        "relations_rejected_unknown_type": 0,
    }
    
    valid_entities = []
    valid_names: Set[str] = set()
    
    for e in entities:
        raw_name = e.get("nom", "")
        clean_name = clean_entity_name(raw_name)
        
        if not clean_name:
            stats["entities_rejected_empty"] += 1
            continue
        if not is_valid_name(clean_name):
            stats["entities_rejected_invalid"] += 1
            continue
        
        e_clean = e.copy()
        e_clean["nom"] = clean_name
        valid_entities.append(e_clean)
        valid_names.add(clean_name)
        stats["entities_kept"] += 1
    
    valid_relations = []
    for r in relations:
        source = clean_entity_name(r.get("source", ""))
        cible = clean_entity_name(r.get("cible", ""))
        rel_type = r.get("type", "").upper()
        
        if rel_type not in VALID_RELATION_TYPES:
            stats["relations_rejected_unknown_type"] += 1
            logger.warning("Type relation inconnu ignoré: %s", rel_type)
            continue
        
        if source not in valid_names or cible not in valid_names:
            stats["relations_rejected_invalid_endpoints"] += 1
            continue
        
        r_clean = r.copy()
        r_clean["source"] = source
        r_clean["cible"] = cible
        r_clean["type"] = rel_type
        valid_relations.append(r_clean)
        stats["relations_kept"] += 1
    
    return valid_entities, valid_relations, stats


def flatten_props(props: dict) -> dict:
    flat = {}
    for k, v in props.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                flat[f"{k}_{sk}"] = sv
        elif isinstance(v, list):
            flat[k] = ", ".join(str(x) for x in v)
        else:
            flat[k] = v
    return flat


def import_to_neo4j(driver, all_valid_entities: List[dict], all_valid_relations: List[dict]) -> Dict[str, int]:
    import_stats = {
        "nodes_created": 0,
        "nodes_merged": 0,
        "relations_created": 0,
        "relations_merged": 0,
    }
    
    with driver.session() as session:
        for e in all_valid_entities:
            label = e["type"]
            if label not in NODE_LABELS:
                logger.warning("Label nœud inconnu: %s", label)
                continue
            
            props = flatten_props(e.get("proprietes", {}))
            props["nom"] = e["nom"]
            
            query = f"""
            MERGE (n:{label} {{nom: $nom}})
            ON CREATE SET n += $props, n.created_at = datetime()
            ON MATCH SET n += $props, n.updated_at = datetime()
            RETURN n
            """
            result = session.run(query, nom=e["nom"], props=props)
            result.consume()
            import_stats["nodes_merged"] += 1
        
        for r in all_valid_relations:
            source_name = r["source"]
            cible_name = r["cible"]
            rel_type = r["type"]
            confiance = r.get("confiance", 1.0)
            a_valider = r.get("a_valider", False)
            
            query = f"""
            MATCH (a {{nom: $source}}), (b {{nom: $cible}})
            MERGE (a)-[rel:{rel_type}]->(b)
            ON CREATE SET rel.confiance = $confiance, rel.a_valider = $a_valider, rel.created_at = datetime()
            ON MATCH SET rel.confiance = $confiance, rel.a_valider = $a_valider, rel.updated_at = datetime()
            RETURN rel
            """
            session.run(query, source=source_name, cible=cible_name, confiance=confiance, a_valider=a_valider)
            import_stats["relations_merged"] += 1
    
    return import_stats


def get_graph_stats(driver) -> Dict:
    stats = {}
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN labels(n)[0] as label, count(*) as count")
        for record in result:
            stats[record["label"]] = record["count"]
        
        result = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(*) as count")
        rel_stats = {}
        for record in result:
            rel_stats[record["type"]] = record["count"]
        stats["relations"] = rel_stats
    
    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Peuplement graphe Neo4j depuis extractions")
    parser.add_argument("--pattern", default="extraction_*.jsonl", help="Pattern fichiers d'extraction")
    args = parser.parse_args()
    
    records = load_all_extractions(args.pattern)
    if not records:
        return
    
    all_valid_entities = []
    all_valid_relations = []
    total_stats = {
        "entities_total": 0,
        "entities_kept": 0,
        "entities_rejected_empty": 0,
        "entities_rejected_invalid": 0,
        "relations_total": 0,
        "relations_kept": 0,
        "relations_rejected_invalid_endpoints": 0,
        "relations_rejected_unknown_type": 0,
    }
    
    for rec in records:
        entities = rec.get("entites", [])
        relations = rec.get("relations", [])
        
        valid_e, valid_r, stats = filter_entities_and_relations(entities, relations)
        all_valid_entities.extend(valid_e)
        all_valid_relations.extend(valid_r)
        
        for k, v in stats.items():
            total_stats[k] += v
    
    logger.info("=== FILTRAGE ===")
    logger.info("Entités: total=%d, gardées=%d, rejetées_vide=%d, rejetées_invalides=%d",
                total_stats["entities_total"], total_stats["entities_kept"],
                total_stats["entities_rejected_empty"], total_stats["entities_rejected_invalid"])
    logger.info("Relations: total=%d, gardées=%d, rejetées_endpoints=%d, rejetées_type=%d",
                total_stats["relations_total"], total_stats["relations_kept"],
                total_stats["relations_rejected_invalid_endpoints"], total_stats["relations_rejected_unknown_type"])
    
    if not all_valid_entities and not all_valid_relations:
        logger.warning("Rien à importer après nettoyage")
        return
    
    driver = get_driver()
    try:
        import_stats = import_to_neo4j(driver, all_valid_entities, all_valid_relations)
        logger.info("=== IMPORT NEO4J ===")
        logger.info("Nœuds créés: %d, fusionnés: %d", import_stats["nodes_created"], import_stats["nodes_merged"])
        logger.info("Relations créées: %d", import_stats["relations_merged"])
        
        graph_stats = get_graph_stats(driver)
        logger.info("=== ÉTAT GRAPHE ===")
        for label, count in graph_stats.items():
            if label != "relations":
                logger.info("  %s: %d", label, count)
        logger.info("  Relations:")
        for rtype, count in graph_stats.get("relations", {}).items():
            logger.info("    %s: %d", rtype, count)
        
    finally:
        driver.close()


if __name__ == "__main__":
    main()