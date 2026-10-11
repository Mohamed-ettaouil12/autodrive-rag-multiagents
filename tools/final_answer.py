#!/usr/bin/env python3
"""
Final Answer Tool - Terminates the ReAct loop and returns the final response.
"""

import logging
from typing import List, Optional, Any
from pydantic import BaseModel, Field

LOG_FORMAT = "%(asctime)s [FINAL_ANSWER] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("final_answer")


class FinalAnswerInput(BaseModel):
    """Input for final_answer tool."""
    answer: str = Field(description="Réponse finale à l'utilisateur")
    sources: List[str] = Field(default_factory=list, description="Sources/documents utilisés")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confiance dans la réponse")
    abstention_reason: Optional[str] = Field(None, description="Raison si abstention")


class FinalAnswerResult(BaseModel):
    """Result - marks the ReAct loop as finished."""
    finished: bool = True
    answer: str
    sources: List[str]
    confidence: float
    abstention_reason: Optional[str] = None


def final_answer(input_data: FinalAnswerInput) -> FinalAnswerResult:
    """
    Termine la boucle ReAct et retourne la réponse finale.
    """
    logger.info("Final answer générée (confidence=%.2f, sources=%d)", 
                input_data.confidence, len(input_data.sources))
    
    return FinalAnswerResult(
        finished=True,
        answer=input_data.answer,
        sources=input_data.sources,
        confidence=input_data.confidence,
        abstention_reason=input_data.abstention_reason,
    )


if __name__ == "__main__":
    # Test
    result = final_answer(FinalAnswerInput(
        answer="La franchise Basic est de 8 000 à 15 000 MAD.",
        sources=["assurances_final.jsonl"],
        confidence=0.95,
    ))
    print(result.model_dump_json(indent=2))