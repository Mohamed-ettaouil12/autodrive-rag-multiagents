import logging
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

LOG_FORMAT = "%(asctime)s [GRAPH] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("schema_graph")


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


CONSTRAINTS = [
    "CREATE CONSTRAINT contrat_numero_unique IF NOT EXISTS FOR (c:Contrat) REQUIRE c.numero IS UNIQUE",
    "CREATE CONSTRAINT vehicule_immatriculation_unique IF NOT EXISTS FOR (v:Vehicule) REQUIRE v.immatriculation IS UNIQUE",
    "CREATE CONSTRAINT assurance_nom_unique IF NOT EXISTS FOR (a:Assurance) REQUIRE a.nom IS UNIQUE",
    "CREATE CONSTRAINT client_id_unique IF NOT EXISTS FOR (c:Client) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT politique_titre_unique IF NOT EXISTS FOR (p:Politique) REQUIRE p.titre IS UNIQUE",
    "CREATE CONSTRAINT categorie_nom_unique IF NOT EXISTS FOR (c:Categorie) REQUIRE c.nom IS UNIQUE",
]


def create_constraints():
    driver = get_driver()
    try:
        with driver.session() as session:
            for constraint in CONSTRAINTS:
                logger.info("Création de la contrainte: %s", constraint)
                session.run(constraint)
        logger.info("Toutes les contraintes ont été créées avec succès")
    except Exception as e:
        logger.error("Erreur lors de la création des contraintes: %s", e)
        raise
    finally:
        driver.close()


def show_constraints():
    driver = get_driver()
    try:
        with driver.session() as session:
            result = session.run("SHOW CONSTRAINTS")
            constraints = list(result)
            logger.info("=== CONTRAINTES ACTUELLES ===")
            for c in constraints:
                logger.info("Contrainte: %s", dict(c))
            return constraints
    finally:
        driver.close()


if __name__ == "__main__":
    print("\n===============================")
    print(" Création du schéma graphe métier")
    print("===============================\n")

    create_constraints()
    print()
    show_constraints()
    print("\n✅ Schéma créé avec succès")