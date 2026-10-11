#!/usr/bin/env python3
"""
Claim Checker - Vérifie chaque affirmation de la réponse contre les preuves récupérées.
LLM-judge avec Mistral, température 0, verdict Pydantic.
"""

import logging
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict

from config import CLAIM_CHECKER_TEMPERATURE, MAX_CLAIM_RETRIES
from utils.structured_output import call_llm_structured, StructuredOutputError

LOG_FORMAT = "%(asctime)s [CLAIM_CHECKER] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("claim_checker")


class ClaimVerdict(BaseModel):
    """Verdict for a single claim."""
    claim: str = Field(description="L'affirmation extraite de la réponse")
    supported: bool = Field(description="True si l'affirmation est soutenue par les preuves")
    evidence: List[str] = Field(default_factory=list, description="Extraits des contextes qui soutiennent l'affirmation")
    confidence: float = Field(ge=0.0, le=1.0, default=1.0, description="Confiance du verdict")


class ClaimCheckResult(BaseModel):
    """Result of claim checking."""
    all_supported: bool
    claims: List[ClaimVerdict]
    unsupported_claims: List[str]


# Prompt for extracting claims from response
EXTRACT_CLAIMS_PROMPT = """Extrais les affirmations factuelles vérifiables de cette réponse.
Une affirmation = une phrase qui fait une déclaration factuelle (chiffre, fait, règle, etc.).
Ignore les phrases de politesse, transitions, "selon le contexte", etc.

RÉPONSE:
{response}

Retourne UNIQUEMENT une liste JSON d'affirmations:
["affirmation 1", "affirmation 2", ...]"""


# Prompt for verifying a single claim
VERIFY_CLAIM_PROMPT = """Tu es un vérificateur factuel. Cette affirmation est-elle SOUTENUE par les preuves ?

AFFIRMATION:
{claim}

PREUVES (contextes récupérés):
{evidence}

RÈGLES:
- "soutenue" = l'information est EXPLICITEMENT dans les preuves
- "non soutenue" = information absente, contradictoire, ou extrapolée
- Sois strict : pas d'inférence, pas de connaissance externe

RÉPONDS UNIQUEMENT EN JSON:
{{
  "supported": true/false,
  "evidence": ["extrait 1", "extrait 2"],
  "confidence": 0.0-1.0
}}"""


def extract_claims(response: str) -> List[str]:
    """Extract verifiable claims from response using LLM."""
    try:
        # Simple heuristic: split by sentences, filter factual ones
        import re
        sentences = re.split(r'[.!?]+', response)
        claims = []
        for s in sentences:
            s = s.strip()
            if len(s) > 20 and any(kw in s.lower() for kw in ['est', 'sont', 'coûte', 'coût', 'montant', 'seuil', 'franchise', 'caution', 'tarif', 'pénalité', 'penalite', 'km', 'mois', 'ans', 'ans', '%', 'mad', 'MAD', 'euro', '€']):
                claims.append(s)
        return claims[:10]  # Max 10 claims
    except Exception as e:
        logger.warning("Claim extraction failed: %s", e)
        return []


def verify_claim(claim: str, evidence: List[str]) -> ClaimVerdict:
    """Verify a single claim against evidence."""
    evidence_str = "\n\n---\n\n".join(evidence[:5]) if evidence else "Aucune preuve"
    
    try:
        verdict = call_llm_structured(
            prompt=VERIFY_CLAIM_PROMPT.format(claim=claim, evidence=evidence_str),
            schema=ClaimVerdict,
            model="mistral",
            temperature=CLAIM_CHECKER_TEMPERATURE,
            max_retries=MAX_CLAIM_RETRIES,
        )
        return verdict
    except StructuredOutputError as e:
        logger.error("Claim verification failed for '%s': %s", claim[:50], e)
        return ClaimVerdict(claim=claim, supported=False, evidence=[], confidence=0.0)


def check_response_claims(response: str, contexts: List[str]) -> ClaimCheckResult:
    """
    Main function: check all claims in response against contexts.
    Returns ClaimCheckResult with all_supported flag.
    """
    if not response or not contexts:
        return ClaimCheckResult(all_supported=True, claims=[], unsupported_claims=[])
    
    # Extract claims
    claims = extract_claims(response)
    if not claims:
        logger.info("No verifiable claims found in response")
        return ClaimCheckResult(all_supported=True, claims=[], unsupported_claims=[])
    
    logger.info("Checking %d claims against %d contexts", len(claims), len(contexts))
    
    # Verify each claim
    verdicts = []
    unsupported = []
    
    for claim in claims:
        verdict = verify_claim(claim, contexts)
        verdicts.append(verdict)
        if not verdict.supported:
            unsupported.append(claim)
            logger.warning("Claim NOT supported: %s", claim[:100])
        else:
            logger.info("Claim supported: %s", claim[:100])
    
    all_supported = len(unsupported) == 0
    
    return ClaimCheckResult(
        all_supported=all_supported,
        claims=verdicts,
        unsupported_claims=unsupported,
    )


def check_and_regenerate(response: str, contexts: List[str], question: str, max_regenerations: int = 1) -> tuple[str, ClaimCheckResult]:
    """
    Check claims, regenerate once if failed, then return final response + check result.
    """
    for attempt in range(max_regenerations + 1):
        result = check_response_claims(response, contexts)
        
        if result.all_supported:
            logger.info("All claims supported on attempt %d", attempt + 1)
            return response, result
        
        if attempt < max_regenerations:
            # Regenerate with feedback
            unsupported_str = "\n".join(f"- {c}" for c in result.unsupported_claims)
            feedback = f"""La réponse précédente contenait des affirmations non soutenues par les preuves:
{unsupported_str}

Régénère la réponse en utilisant UNIQUEMENT les informations des contextes fournis.
Si une information n'est pas dans les contextes, ne l'inclus pas."""
            
            try:
                from rag_autodrive import generate_answer
                # Regenerate with feedback
                context_str = "\n\n---\n\n".join(contexts[:5])
                prompt = f"""Tu es un assistant AutoDrive Maroc. Réponds en utilisant UNIQUEMENT le contexte.

CONTEXTE:
{context_str}

QUESTION:
{question}

FEEDBACK PRÉCÉDENT:
{feedback}

RÉPONSE:"""
                
                response = generate_answer(question, contexts)
                logger.info("Regenerated response (attempt %d)", attempt + 2)
            except Exception as e:
                logger.error("Regeneration failed: %s", e)
                break
        else:
            # Max regenerations reached - add abstention note
            unsupported_str = "\n".join(f"- {c}" for c in result.unsupported_claims)
            response += f"\n\n[Note: Les affirmations suivantes n'ont pas pu être vérifiées: {unsupported_str}]"
    
    return response, result


if __name__ == "__main__":
    # Test
    test_response = "La franchise Basic est de 10 000 MAD. Le tarif journalier est 220 MAD."
    test_contexts = ["Franchise Basic: 8000-15000 MAD. Tarif journalier catégorie A: 220 MAD/jour."]
    
    result = check_response_claims(test_response, test_contexts)
    print(f"All supported: {result.all_supported}")
    for v in result.claims:
        print(f"  {v.claim} -> supported={v.supported}, conf={v.confidence}")