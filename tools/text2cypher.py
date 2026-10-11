#!/usr/bin/env python3
"""
Text2Cypher Tool - Wrapper for graph_query.py with structured output.
Called by ReAct agent when it needs graph data.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from graph_query import (
    get_contrat,
    get_assurance_contrat,
    get_historique_vehicule,
    recherche_entite,
    is_neo4j_available,
)

LOG_FORMAT = "%(asctime)s [TEXT2CYPHER] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("text2cypher")


class Text2CypherInput(BaseModel):
    """Input schema for text2cypher tool."""
    query_type: str = Field(description="Type de requête: 'contrat', 'assurance_contrat', 'historique_vehicule', 'recherche_entite'")
    identifier: str = Field(description="Numéro contrat, immatriculation, ou nom partiel selon query_type")
    

class Text2CypherResult(BaseModel):
    """Result schema for text2cypher tool."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    query_type: str
    identifier: str


def text2cypher(input_data: Text2CypherInput) -> Text2CypherResult:
    """
    Execute a graph query based on the input type and identifier.
    """
    logger.info("Text2Cypher called: type=%s, identifier=%s", input_data.query_type, input_data.identifier)
    
    if not is_neo4j_available():
        return Text2CypherResult(
            success=False,
            error="Neo4j non disponible",
            query_type=input_data.query_type,
            identifier=input_data.identifier,
        )
    
    try:
        if input_data.query_type == "contrat":
            data = get_contrat(input_data.identifier)
        elif input_data.query_type == "assurance_contrat":
            data = get_assurance_contrat(input_data.identifier)
        elif input_data.query_type == "historique_vehicule":
            data = get_historique_vehicule(input_data.identifier)
        elif input_data.query_type == "recherche_entite":
            data = recherche_entite(input_data.identifier)
        else:
            return Text2CypherResult(
                success=False,
                error=f"Type de requête inconnu: {input_data.query_type}",
                query_type=input_data.query_type,
                identifier=input_data.identifier,
            )
        
        return Text2CypherResult(
            success=True,
            data=data,
            query_type=input_data.query_type,
            identifier=input_data.identifier,
        )
    except Exception as e:
        logger.error("Erreur text2cypher: %s", e)
        return Text2CypherResult(
            success=False,
            error=str(e),
            query_type=input_data.query_type,
            identifier=input_data.identifier,
        )


if __name__ == "__main__":
    # Test
    test_cases = [
        Text2CypherInput(query_type="contrat", identifier="CASA-2024-014"),
        Text2CypherInput(query_type="assurance_contrat", identifier="CASA-2024-014"),
        Text2CypherInput(query_type="historique_vehicule", identifier="ADM-A-CASABLANCA-01-001"),
        Text2CypherInput(query_type="recherche_entite", identifier="CASA"),
    ]
    
    for tc in test_cases:
        result = text2cypher(tc)
        print(f"\n{tc.query_type}({tc.identifier}): success={result.success}")
        if result.data:
            print(json.dumps(result.data, indent=2, ensure_ascii=False)[:500])
        if result.error:
            print(f"  Error: {result.error}")