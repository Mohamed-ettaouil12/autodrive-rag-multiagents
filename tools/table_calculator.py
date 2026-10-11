#!/usr/bin/env python3
"""
Table Calculator Tool - Deterministic numerical extraction from entity chunks.
Loads structured entity data (contrats) and parses numerical values from text chunks (assurances, maintenance).
The LLM calls this tool and reformulates the result - it NEVER calculates itself.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

LOG_FORMAT = "%(asctime)s [TABLE_CALC] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("table_calculator")

# Paths
DATA_DIR = Path("/home/tawil/Bureau/stage d'ete/preprocessing/output")

# Entity fields mapping for contrats (from chunks_contrats.jsonl entities)
CONTRAT_ENTITY_FIELDS = {
    "applicable_rate_mad": "entity_applicable_rate_mad",
    "deposit_mad": "entity_deposit_mad",
    "included_km_per_day": "entity_included_mileage_rule",
    "extra_km_rate_mad": "entity_included_mileage_rule",
    "late_return_tolerance_h": "entity_late_return_rule",
    "late_return_penalty_pct": "entity_late_return_rule",
    "late_return_full_day_threshold_h": "entity_late_return_rule",
    "fuel_service_fee_mad": "entity_fuel_rule",
}

# Insurance franchise patterns in FAQ chunks
INSURANCE_FRANCHISE_PATTERNS = {
    "Basic": [
        r"franchise.*?(\d[\d\s]*)\s*MAD",
        r"franchise.*?de\s+(\d[\d\s]*)\s*MAD",
        r"franchise\s+(?:de\s+)?(\d[\d\s]*)\s*MAD",
        r"(\d[\d\s]*)\s*MAD.*?franchise",
    ],
    "CDW Standard": [
        r"franchise.*?(\d[\d\s]*)\s*MAD",
        r"franchise.*?selon contrat",
    ],
    "CDW Premium": [
        r"franchise.*?(\d[\d\s]*)\s*MAD",
        r"comprise entre\s+(\d[\d\s]*)\s*et\s+(\d[\d\s]*)\s*MAD",
        r"entre\s+(\d[\d\s]*)\s*et\s+(\d[\d\s]*)\s*MAD",
        r"(\d[\d\s]*)\s*à\s*(\d[\d\s]*)\s*MAD.*?CDW Premium",
        r"CDW Premium.*?(\d[\d\s]*)\s*à\s*(\d[\d\s]*)\s*MAD",
    ],
    "Full Protection": [
        r"franchise.*?(\d[\d\s]*)\s*MAD",
        r"comprise entre\s+(\d[\d\s]*)\s*et\s+(\d[\d\s]*)\s*MAD",
        r"entre\s+(\d[\d\s]*)\s*et\s+(\d[\d\s]*)\s*MAD",
        r"(\d[\d\s]*)\s*à\s*(\d[\d\s]*)\s*MAD.*?Full Protection",
        r"Full Protection.*?(\d[\d\s]*)\s*à\s*(\d[\d\s]*)\s*MAD",
    ],
}

# Maintenance threshold patterns
MAINTENANCE_PATTERNS = {
    "vidange_km": r"vidange.*?(\d[\d\s]*)\s*km",
    "pneus_km": r"pneus.*?(\d[\d\s]*)\s*km",
    "freinage_km": r"freinage.*?(\d[\d\s]*)\s*km",
    "batterie_mois": r"batterie.*?(\d+)\s*mois",
    "climatisation_mois": r"climatisation.*?(\d+)\s*mois",
}


@dataclass
class TableCalcResult:
    value: float
    unit: str
    source_chunks: List[str]
    formula: str
    raw_value: Optional[str] = None


class TableCalculator:
    """Deterministic calculator for numerical values from document chunks."""
    
    def __init__(self):
        self.contrat_entities: Dict[str, Dict] = {}  # contract_number -> entity fields
        self.assurance_faq_chunks: List[Dict] = []
        self.maintenance_chunks: List[Dict] = []
        self._loaded = False
    
    def _load_all(self):
        """Load all chunk files once."""
        if self._loaded:
            return
        
        # Load contrat entities
        contrats_file = DATA_DIR / "chunks_contrats.jsonl"
        if contrats_file.exists():
            with open(contrats_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if chunk.get("chunk_type") == "entities":
                        meta = chunk.get("metadata", {})
                        contract_num = meta.get("entity_contract_number", "")
                        if contract_num:
                            self.contrat_entities[contract_num] = meta
        
        # Load assurance FAQ chunks for franchise extraction
        assurances_file = DATA_DIR / "chunks_assurances.jsonl"
        if assurances_file.exists():
            with open(assurances_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if chunk.get("chunk_type") in ("faq", "assurance_exclusion", "entities"):
                        self.assurance_faq_chunks.append(chunk)
        
        # Load maintenance chunks
        maintenance_file = DATA_DIR / "chunks_maintenance.jsonl"
        if maintenance_file.exists():
            with open(maintenance_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    chunk = json.loads(line)
                    self.maintenance_chunks.append(chunk)
        
        self._loaded = True
        logger.info(f"Loaded {len(self.contrat_entities)} contrat entities, "
                   f"{len(self.assurance_faq_chunks)} assurance chunks, "
                   f"{len(self.maintenance_chunks)} maintenance chunks")
    
    def calculate(self, question: str, category: str) -> Optional[TableCalcResult]:
        """Main entry point - routes to appropriate calculator based on category/question."""
        self._load_all()
        
        question_lower = question.lower()
        
        if category == "contrats":
            return self._calculate_contrats(question_lower)
        elif category == "assurances":
            return self._calculate_assurances(question_lower)
        elif category == "maintenance":
            return self._calculate_maintenance(question_lower)
        
        return None
    
    def _calculate_contrats(self, question: str) -> Optional[TableCalcResult]:
        """Calculate contrat numerical values from entity chunks."""
        
        # Extract contract number if present
        contract_match = re.search(r"[A-Z]{2,6}-\d{4}-\d{2,3}", question)
        contract_num = contract_match.group(0) if contract_match else None
        
        # Find relevant entity data
        entity_data = None
        if contract_num and contract_num in self.contrat_entities:
            entity_data = self.contrat_entities[contract_num]
        else:
            # Fallback: use first available entity (for general category questions)
            if self.contrat_entities:
                entity_data = list(self.contrat_entities.values())[0]
        
        if not entity_data:
            return None
        
        # Check specific keywords FIRST (more specific) before general ones
        
        # Kilométrage inclus / km supplémentaire
        if any(kw in question for kw in ["kilométrage", "kilometrage", "km", "kilomètre"]):
            mileage_rule = entity_data.get("entity_included_mileage_rule", "")
            # Parse "200 km/jour inclus, 2.5 MAD par km supplémentaire"
            incl_match = re.search(r"(\d+)\s*km", mileage_rule)
            extra_match = re.search(r"(\d+\.?\d*)\s*MAD\s*par\s*km", mileage_rule)
            
            if any(kw in question for kw in ["supplémentaire", "supplementaire", "excédent", "excedent", "additional", "extra"]):
                if extra_match:
                    return TableCalcResult(
                        value=float(extra_match.group(1).replace(",", ".")),
                        unit="MAD/km",
                        source_chunks=[k for k, v in self.contrat_entities.items() if v == entity_data][:1],
                        formula=f"Parsed from entity_included_mileage_rule: {extra_match.group(1)} MAD/km supplémentaire"
                    )
            else:
                if incl_match:
                    return TableCalcResult(
                        value=float(incl_match.group(1)),
                        unit="km/jour",
                        source_chunks=[k for k, v in self.contrat_entities.items() if v == entity_data][:1],
                        formula=f"Parsed from entity_included_mileage_rule: {incl_match.group(1)} km/jour inclus"
                    )
        
        # Caution / dépôt
        if any(kw in question for kw in ["caution", "dépôt", "depot", "garantie"]):
            deposit = entity_data.get("entity_deposit_mad")
            if deposit is not None:
                return TableCalcResult(
                    value=float(deposit),
                    unit="MAD",
                    source_chunks=[k for k, v in self.contrat_entities.items() if v == entity_data][:1],
                    formula=f"entity_deposit_mad = {deposit}"
                )
        
        # Pénalité retard
        if any(kw in question for kw in ["retard", "pénalité", "penalite", "penalité", "tolérance"]):
            late_rule = entity_data.get("entity_late_return_rule", "")
            # Parse "1 heure de tolérance ; au-delà, 10% du tarif journalier facturé par heure de retard ; au-delà de 3 heures, une journée supplémentaire complète est facturée"
            pct_match = re.search(r"(\d+)%", late_rule)
            tolerance_match = re.search(r"(\d+)\s*heure.*?tolérance", late_rule)
            threshold_match = re.search(r"au-delà de (\d+)\s*heures?", late_rule)
            
            if pct_match:
                return TableCalcResult(
                    value=float(pct_match.group(1)),
                    unit="%/heure",
                    source_chunks=[k for k, v in self.contrat_entities.items() if v == entity_data][:1],
                    formula=f"Parsed from entity_late_return_rule: {pct_match.group(1)}% du tarif journalier par heure de retard"
                )
        
        # Frais carburant
        if any(kw in question for kw in ["carburant", "réservoir", "reservoir", "plein", "frais de service"]):
            fuel_rule = entity_data.get("entity_fuel_rule", "")
            fee_match = re.search(r"(\d+)\s*MAD\s*de\s*frais", fuel_rule)
            if fee_match:
                return TableCalcResult(
                    value=float(fee_match.group(1)),
                    unit="MAD",
                    source_chunks=[k for k, v in self.contrat_entities.items() if v == entity_data][:1],
                    formula=f"Parsed from entity_fuel_rule: {fee_match.group(1)} MAD frais de service"
                )
        
        # Tarif journalier / mensuel (GENERAL - check last)
        if any(kw in question for kw in ["tarif", "prix", "coût", "combien", "journalier", "mensuel", "quotidien"]):
            rate = entity_data.get("entity_applicable_rate_mad")
            if rate is not None:
                basis = entity_data.get("entity_rental_basis", "daily")
                unit = "MAD/jour" if basis == "daily" else "MAD/mois"
                return TableCalcResult(
                    value=float(rate),
                    unit=unit,
                    source_chunks=[k for k, v in self.contrat_entities.items() if v == entity_data][:1],
                    formula=f"entity_applicable_rate_mad = {rate} ({basis})"
                )
        
        return None
    
    def _calculate_assurances(self, question: str) -> Optional[TableCalcResult]:
        """Extract franchise values from assurance FAQ chunks."""
        
        # Identify insurance type from question
        insurance_type = None
        for itype in ["Basic", "CDW Standard", "CDW Premium", "Full Protection"]:
            if itype.lower() in question:
                insurance_type = itype
                break
        
        if not insurance_type:
            # Try to infer from context
            if "premium" in question:
                insurance_type = "CDW Premium"
            elif "full" in question or "protection" in question:
                insurance_type = "Full Protection"
            elif "standard" in question:
                insurance_type = "CDW Standard"
            else:
                insurance_type = "Basic"
        
        # Search in FAQ chunks for franchise info
        patterns = INSURANCE_FRANCHISE_PATTERNS.get(insurance_type, [])
        
        for chunk in self.assurance_faq_chunks:
            text = chunk.get("text", "")
            meta = chunk.get("metadata", {})
            chunk_insurance = meta.get("insurance_type", "")
            
            # Match insurance type
            if chunk_insurance and chunk_insurance != insurance_type:
                continue
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if len(match.groups()) == 2:  # Range (min-max)
                        min_val = float(match.group(1).replace(" ", "").replace(",", "."))
                        max_val = float(match.group(2).replace(" ", "").replace(",", "."))
                        return TableCalcResult(
                            value=(min_val + max_val) / 2,  # Return average
                            unit="MAD",
                            source_chunks=[chunk.get("chunk_id", "")],
                            formula=f"Franchise {insurance_type}: {min_val}-{max_val} MAD (moyenne: {(min_val+max_val)/2:.0f} MAD)",
                            raw_value=f"{min_val}-{max_val} MAD"
                        )
                    else:  # Single value
                        val = float(match.group(1).replace(" ", "").replace(",", "."))
                        return TableCalcResult(
                            value=val,
                            unit="MAD",
                            source_chunks=[chunk.get("chunk_id", "")],
                            formula=f"Franchise {insurance_type}: {val} MAD",
                            raw_value=f"{val} MAD"
                        )
        
        return None
    
    def _calculate_maintenance(self, question: str) -> Optional[TableCalcResult]:
        """Extract maintenance thresholds from chunks."""
        
        # Identify category
        category = None
        for cat in ["A", "B", "C", "D", "E", "F"]:
            if f"catégorie {cat}" in question or f"categorie {cat}" in question or f"cat {cat}" in question:
                category = cat
                break
        
        for chunk in self.maintenance_chunks:
            text = chunk.get("text", "")
            meta = chunk.get("metadata", {})
            chunk_cat = meta.get("vehicle_category", "")
            
            if category and chunk_cat and chunk_cat != category:
                continue
            
            text_lower = text.lower()
            
            # Vidange
            if "vidange" in question:
                match = re.search(r"vidange.*?(\d[\d\s]*)\s*km", text_lower)
                if match:
                    val = float(match.group(1).replace(" ", ""))
                    return TableCalcResult(
                        value=val,
                        unit="km",
                        source_chunks=[chunk.get("chunk_id", "")],
                        formula=f"Seuil vidange catégorie {category or chunk_cat or '?'}: {val} km"
                    )
            
            # Pneus et freinage (both)
            if "pneu" in question and "frein" in question:
                pneu_match = re.search(r"pneus.*?(\d[\d\s]*)\s*km", text_lower)
                frein_match = re.search(r"freinage.*?(\d[\d\s]*)\s*km", text_lower)
                if pneu_match and frein_match:
                    pneu_val = float(pneu_match.group(1).replace(" ", ""))
                    frein_val = float(frein_match.group(1).replace(" ", ""))
                    return TableCalcResult(
                        value=pneu_val,  # Return first value, formula explains both
                        unit="km",
                        source_chunks=[chunk.get("chunk_id", "")],
                        formula=f"Seuils catégorie {category or chunk_cat or '?'}: pneus={pneu_val} km, freinage={frein_val} km"
                    )
            
            # Pneus
            if "pneu" in question:
                match = re.search(r"pneus.*?(\d[\d\s]*)\s*km", text_lower)
                if match:
                    val = float(match.group(1).replace(" ", ""))
                    return TableCalcResult(
                        value=val,
                        unit="km",
                        source_chunks=[chunk.get("chunk_id", "")],
                        formula=f"Seuil pneus catégorie {category or chunk_cat or '?'}: {val} km"
                    )
            
            # Freinage
            if "frein" in question:
                match = re.search(r"freinage.*?(\d[\d\s]*)\s*km", text_lower)
                if match:
                    val = float(match.group(1).replace(" ", ""))
                    return TableCalcResult(
                        value=val,
                        unit="km",
                        source_chunks=[chunk.get("chunk_id", "")],
                        formula=f"Seuil freinage catégorie {category or chunk_cat or '?'}: {val} km"
                    )
            
            # Batterie
            if "batterie" in question:
                match = re.search(r"batterie.*?(\d+)\s*mois", text_lower)
                if match:
                    val = float(match.group(1))
                    return TableCalcResult(
                        value=val,
                        unit="mois",
                        source_chunks=[chunk.get("chunk_id", "")],
                        formula=f"Seuil batterie catégorie {category or chunk_cat or '?'}: {val} mois"
                    )
            
            # Climatisation
            if "climatisation" in question or "clim" in question:
                match = re.search(r"climatisation.*?(\d+)\s*mois", text_lower)
                if match:
                    val = float(match.group(1))
                    return TableCalcResult(
                        value=val,
                        unit="mois",
                        source_chunks=[chunk.get("chunk_id", "")],
                        formula=f"Seuil climatisation catégorie {category or chunk_cat or '?'}: {val} mois"
                    )
        
        return None


# Singleton instance
_calculator = None

def get_calculator() -> TableCalculator:
    global _calculator
    if _calculator is None:
        _calculator = TableCalculator()
    return _calculator


def calculate_table(question: str, category: str) -> Optional[TableCalcResult]:
    """Public API function."""
    calc = get_calculator()
    return calc.calculate(question, category)


if __name__ == "__main__":
    # Test
    calc = TableCalculator()
    
    test_questions = [
        ("Quel est le tarif journalier du contrat CASA-2024-014 ?", "contrats"),
        ("Quelle est la caution pour le contrat CASA-2024-014 ?", "contrats"),
        ("Combien coûte le km supplémentaire pour le contrat CASA-2024-014 ?", "contrats"),
        ("Quel est le pourcentage de pénalité par heure de retard ?", "contrats"),
        ("Quel est le montant des frais de service carburant ?", "contrats"),
        ("Quelle est la franchise de l'assurance Basic ?", "assurances"),
        ("Quelle est la franchise de la CDW Premium ?", "assurances"),
        ("Quelle est la franchise de la Full Protection ?", "assurances"),
        ("Quel est le seuil de vidange pour la catégorie A ?", "maintenance"),
        ("Quels sont les seuils pneus et freinage pour la catégorie A ?", "maintenance"),
    ]
    
    print("=" * 80)
    print("TABLE CALCULATOR TESTS")
    print("=" * 80)
    
    for q, cat in test_questions:
        result = calc.calculate(q, cat)
        if result:
            print(f"\nQ: {q}")
            print(f"  Category: {cat}")
            print(f"  Value: {result.value} {result.unit}")
            print(f"  Formula: {result.formula}")
            print(f"  Sources: {result.source_chunks}")
        else:
            print(f"\nQ: {q} -> NO RESULT")