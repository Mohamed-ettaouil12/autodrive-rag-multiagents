#!/usr/bin/env python3
"""
Clarification Tool - Asks user for clarification when question is ambiguous.
Returns a structured response that the ReAct loop can use.
"""

import logging
from typing import Optional
from pydantic import BaseModel, Field

LOG_FORMAT = "%(asctime)s [CLARIFICATION] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("clarification")


class ClarificationInput(BaseModel):
    """Input for clarification tool."""
    question: str = Field(description="Question originale de l'utilisateur")
    missing_info: str = Field(description="Information manquante pour répondre précisément")
    suggested_options: list[str] = Field(default_factory=list, description="Options suggérées pour l'utilisateur")


class ClarificationResult(BaseModel):
    """Result from clarification tool - in practice, this would wait for user input.
    For automated evaluation, we return a structured response indicating clarification needed."""
    clarification_needed: bool = True
    question: str
    missing_info: str
    suggested_options: list[str] = Field(default_factory=list)
    message: str = Field(description="Message à afficher à l'utilisateur")


def ask_clarification(input_data: ClarificationInput) -> ClarificationResult:
    """
    Demande une clarification à l'utilisateur.
    Dans le contexte automatisé (évaluation), on retourne une réponse structurée
    indiquant qu'une clarification est nécessaire.
    """
    logger.info("Clarification demandée pour: %s", input_data.question)
    
    options_str = ""
    if input_data.suggested_options:
        options_str = "\nOptions possibles: " + ", ".join(input_data.suggested_options)
    
    message = (
        f"J'ai besoin d'une précision pour répondre correctement.\n"
        f"Question: {input_data.question}\n"
        f"Information manquante: {input_data.missing_info}{options_str}\n"
        f"Pourriez-vous préciser ?"
    )
    
    return ClarificationResult(
        question=input_data.question,
        missing_info=input_data.missing_info,
        suggested_options=input_data.suggested_options,
        message=message,
    )


if __name__ == "__main__":
    # Test
    result = ask_clarification(ClarificationInput(
        question="Quelle est la franchise ?",
        missing_info="Type d'assurance (Basic, CDW Standard, CDW Premium, Full Protection)",
        suggested_options=["Basic", "CDW Standard", "CDW Premium", "Full Protection"],
    ))
    print(result.message)