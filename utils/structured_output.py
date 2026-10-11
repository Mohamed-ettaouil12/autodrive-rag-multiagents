#!/usr/bin/env python3
"""
Utilitaire générique pour sorties structurées LLM (Ollama + Pydantic).
Réutilisable par supervisor, reformulateur, ReAct tools, claim checker, etc.
"""

import json
import logging
from typing import Type, TypeVar, Optional, Any, Dict

import ollama
from pydantic import BaseModel, ValidationError

from config import ENABLE_STRUCTURED_OUTPUT, MAX_JSON_RETRIES, OLLAMA_JSON_FORMAT

T = TypeVar("T", bound=BaseModel)

LOG_FORMAT = "%(asctime)s [STRUCTURED] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("structured_output")

# Global counters for ablation metrics
_structured_output_stats = {
    "total_calls": 0,
    "validation_failures": 0,
    "retries_used": 0,
    "final_failures": 0,
}


def reset_structured_output_stats():
    """Reset counters for a new evaluation run."""
    global _structured_output_stats
    _structured_output_stats = {
        "total_calls": 0,
        "validation_failures": 0,
        "retries_used": 0,
        "final_failures": 0,
    }


def get_structured_output_stats() -> Dict[str, int]:
    """Get current stats."""
    return _structured_output_stats.copy()


class StructuredOutputError(Exception):
    """Erreur lors de la génération/validation de sortie structurée."""
    pass


def _clean_json_raw(raw: str) -> str:
    """Nettoie la réponse brute (retire blocs markdown ```json ... ```)."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    return raw


def _parse_and_validate(raw: str, schema: Type[T]) -> Optional[T]:
    """Parse JSON et valide avec Pydantic."""
    raw = _clean_json_raw(raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.debug("Erreur parsing JSON: %s | raw: %s", e, raw[:200])
        return None

    try:
        return schema.model_validate(data)
    except ValidationError as e:
        logger.debug("Validation Pydantic échouée: %s | raw: %s", e, raw[:200])
        return None


def call_llm_structured(
    prompt: str,
    schema: Type[T],
    model: str = "mistral",
    temperature: float = 0.0,
    system_prompt: Optional[str] = None,
    max_retries: int = MAX_JSON_RETRIES,
    use_json_format: bool = OLLAMA_JSON_FORMAT,
    **ollama_kwargs: Any,
) -> T:
    """
    Appelle Ollama avec décodage contraint JSON + validation Pydantic + retry borné.
    
    Args:
        prompt: Prompt utilisateur
        schema: Classe Pydantic pour validation
        model: Modèle Ollama (défaut: mistral)
        temperature: Température (défaut: 0.0 pour déterminisme)
        system_prompt: Prompt système optionnel
        max_retries: Nombre max de tentatives (défaut: config.MAX_JSON_RETRIES)
        use_json_format: Utiliser format="json" Ollama (défaut: config.OLLAMA_JSON_FORMAT)
        **ollama_kwargs: Arguments supplémentaires pour ollama.chat()
    
    Returns:
        Instance validée du schema Pydantic
    
    Raises:
        StructuredOutputError: Si échec après tous les retries
    """
    global _structured_output_stats
    _structured_output_stats["total_calls"] += 1
    
    if not ENABLE_STRUCTURED_OUTPUT:
        # Fallback: appel simple sans validation structurée
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = ollama.chat(model=model, messages=messages, options={"temperature": temperature}, **ollama_kwargs)
        raw = response["message"]["content"]
        validated = _parse_and_validate(raw, schema)
        if validated is None:
            _structured_output_stats["validation_failures"] += 1
            _structured_output_stats["final_failures"] += 1
            raise StructuredOutputError(f"Sortie non structurée invalide (ENABLE_STRUCTURED_OUTPUT=False): {raw[:200]}")
        return validated

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    base_kwargs = {
        "model": model,
        "messages": messages,
        "options": {"temperature": temperature},
        **ollama_kwargs,
    }
    if use_json_format:
        base_kwargs["format"] = "json"

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = ollama.chat(**base_kwargs)
            raw_content = response["message"]["content"]
            logger.debug("Tentative %d/%d - Réponse brute: %s", attempt + 1, max_retries + 1, raw_content[:300])

            validated = _parse_and_validate(raw_content, schema)
            if validated is not None:
                if attempt > 0:
                    _structured_output_stats["retries_used"] += attempt
                logger.info("Sortie structurée validée (tentative %d)", attempt + 1)
                return validated

            _structured_output_stats["validation_failures"] += 1
            last_error = "Validation Pydantic échouée"
        except Exception as e:
            last_error = str(e)
            logger.warning("Erreur appel Ollama (tentative %d): %s", attempt + 1, e)

        if attempt < max_retries:
            logger.warning("Réessai %d/%d...", attempt + 1, max_retries)

    _structured_output_stats["final_failures"] += 1
    raise StructuredOutputError(
        f"Échec après {max_retries + 1} tentatives. Dernière erreur: {last_error}"
    )


def call_llm_structured_with_raw(
    prompt: str,
    schema: Type[T],
    model: str = "mistral",
    temperature: float = 0.0,
    system_prompt: Optional[str] = None,
    max_retries: int = MAX_JSON_RETRIES,
    use_json_format: bool = OLLAMA_JSON_FORMAT,
    **ollama_kwargs: Any,
) -> tuple[T, str]:
    """
    Version qui retourne aussi la réponse brute (pour debug/traces).
    """
    global _structured_output_stats
    _structured_output_stats["total_calls"] += 1
    
    if not ENABLE_STRUCTURED_OUTPUT:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = ollama.chat(model=model, messages=messages, options={"temperature": temperature}, **ollama_kwargs)
        raw = response["message"]["content"]
        validated = _parse_and_validate(raw, schema)
        if validated is None:
            _structured_output_stats["validation_failures"] += 1
            _structured_output_stats["final_failures"] += 1
            raise StructuredOutputError(f"Sortie non structurée invalide: {raw[:200]}")
        return validated, raw

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    base_kwargs = {
        "model": model,
        "messages": messages,
        "options": {"temperature": temperature},
        **ollama_kwargs,
    }
    if use_json_format:
        base_kwargs["format"] = "json"

    last_error = None
    last_raw = ""
    for attempt in range(max_retries + 1):
        try:
            response = ollama.chat(**base_kwargs)
            raw_content = response["message"]["content"]
            last_raw = raw_content
            logger.debug("Tentative %d/%d - Réponse brute: %s", attempt + 1, max_retries + 1, raw_content[:300])

            validated = _parse_and_validate(raw_content, schema)
            if validated is not None:
                if attempt > 0:
                    _structured_output_stats["retries_used"] += attempt
                logger.info("Sortie structurée validée (tentative %d)", attempt + 1)
                return validated, raw_content

            _structured_output_stats["validation_failures"] += 1
            last_error = "Validation Pydantic échouée"
        except Exception as e:
            last_error = str(e)
            logger.warning("Erreur appel Ollama (tentative %d): %s", attempt + 1, e)

        if attempt < max_retries:
            logger.warning("Réessai %d/%d...", attempt + 1, max_retries)

    _structured_output_stats["final_failures"] += 1
    raise StructuredOutputError(
        f"Échec après {max_retries + 1} tentatives. Dernière erreur: {last_error}. Dernière réponse: {last_raw[:200]}"
    )


if __name__ == "__main__":
    # Test rapide
    from schemas import RoutingDecision, ReformulatedQuestion
    
    print("Test call_llm_structured...")
    
    # Test RoutingDecision
    try:
        result = call_llm_structured(
            prompt='Question: "Quelle est la franchise ?"',
            schema=RoutingDecision,
            system_prompt='Tu es un routeur. Réponds UNIQUEMENT en JSON: {"agents":["assurances"]}',
        )
        print(f"RoutingDecision: {result}")
    except StructuredOutputError as e:
        print(f"Erreur: {e}")
    
    # Test ReformulatedQuestion
    try:
        result = call_llm_structured(
            prompt='Historique: User: "Quelle est la franchise ?" Assistant: "Basic: 8000-15000 MAD" Question: "Et pour Premium ?"',
            schema=ReformulatedQuestion,
            system_prompt='Réécris la question pour qu\'elle soit autonome. JSON: {"question": "..."}',
        )
        print(f"ReformulatedQuestion: {result}")
    except StructuredOutputError as e:
        print(f"Erreur: {e}")