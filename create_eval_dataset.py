#!/usr/bin/env python3
"""
Create evaluation dataset from real documents for RAGAS evaluation.
"""

import json
import random
from pathlib import Path

DATASET_DIR = Path("/home/tawil/Bureau/stage d'ete/dataset")
OUTPUT_FILE = Path("/home/tawil/Bureau/stage d'ete/dataset_eval.jsonl")

# Target counts per category
TARGET_COUNTS = {
    "contrats": 15,
    "assurances": 12,
    "faq": 12,
    "vehicules": 11,
    "maintenance": 10,
}

def load_jsonl(filepath):
    """Load JSONL file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def extract_questions_from_politiques_contrats(docs, count):
    """Extract questions from politiques_contrats documents."""
    questions = []
    for doc in docs:
        entities = doc.get("entities", {})
        contract_num = entities.get("contract_number", "")
        agency = entities.get("agency_name", "")
        category = entities.get("vehicle_category_name", "")
        rate = entities.get("applicable_rate_mad", 0)
        deposit = entities.get("deposit_mad", 0)
        age_rule = entities.get("minimum_driver_age_rule", "")
        mileage_rule = entities.get("included_mileage_rule", "")
        late_rule = entities.get("late_return_rule", "")
        fuel_rule = entities.get("fuel_rule", "")
        docs_req = entities.get("required_documents", "")
        
        sections = doc.get("sections", [])
        
        # Contract-specific questions
        if contract_num:
            questions.append({
                "question": f"Quelles sont les pièces requises pour le contrat {contract_num} ?",
                "ground_truth": docs_req,
                "source": f"politiques_contrats.jsonl - {doc.get('document_id', '')}"
            })
            
            questions.append({
                "question": f"Quel est le tarif journalier du contrat {contract_num} ?",
                "ground_truth": f"{rate} MAD",
                "source": f"politiques_contrats.jsonl - {doc.get('document_id', '')}"
            })
            
            questions.append({
                "question": f"Quelle est la règle de retard pour le contrat {contract_num} ?",
                "ground_truth": late_rule,
                "source": f"politiques_contrats.jsonl - {doc.get('document_id', '')}"
            })
            
            questions.append({
                "question": f"Quel est l'âge minimum pour le contrat {contract_num} ?",
                "ground_truth": age_rule,
                "source": f"politiques_contrats.jsonl - {doc.get('document_id', '')}"
            })
            
            questions.append({
                "question": f"Quelle est la caution pour le contrat {contract_num} ?",
                "ground_truth": f"{deposit} MAD",
                "source": f"politiques_contrats.jsonl - {doc.get('document_id', '')}"
            })
        
        # Section-based questions
        for section in sections:
            content = section.get("content", "")
            title = section.get("title", "")
            if "caution" in title.lower() and deposit:
                questions.append({
                    "question": f"Comment est calculée la caution pour un contrat {category} à {agency} ?",
                    "ground_truth": content[:500],
                    "source": f"politiques_contrats.jsonl - {doc.get('document_id', '')}"
                })
            if "kilométrage" in title.lower() and mileage_rule:
                questions.append({
                    "question": f"Quel est le kilométrage inclus pour le contrat {contract_num} ?",
                    "ground_truth": mileage_rule,
                    "source": f"politiques_contrats.jsonl - {doc.get('document_id', '')}"
                })
            if "carburant" in title.lower() and fuel_rule:
                questions.append({
                    "question": f"Quelle est la règle carburant pour le contrat {contract_num} ?",
                    "ground_truth": fuel_rule,
                    "source": f"politiques_contrats.jsonl - {doc.get('document_id', '')}"
                })
            if "annulation" in title.lower():
                questions.append({
                    "question": f"Quelle est la politique d'annulation pour {contract_num} ?",
                    "ground_truth": content[:500],
                    "source": f"politiques_contrats.jsonl - {doc.get('document_id', '')}"
                })
            if "paiement" in title.lower():
                questions.append({
                    "question": f"Quelles sont les modalités de paiement pour {contract_num} ?",
                    "ground_truth": content[:500],
                    "source": f"politiques_contrats.jsonl - {doc.get('document_id', '')}"
                })
    
    return questions[:count]

def extract_questions_from_assurances(docs, count):
    """Extract questions from assurances documents."""
    questions = []
    for doc in docs:
        insurance_type = doc.get("insurance_type", "")
        insurance_code = doc.get("insurance_code", "")
        covers = doc.get("covers", [])
        exclusions = doc.get("exclusions", [])
        sections = doc.get("sections", [])
        
        franchise = None
        for section in sections:
            if "franchise" in section.get("title", "").lower():
                franchise = section.get("content", "")
                break
        
        if covers:
            questions.append({
                "question": f"Quelles sont les couvertures de l'assurance {insurance_type} ?",
                "ground_truth": ", ".join(covers),
                "source": f"assurances_final.jsonl - {doc.get('document_id', '')}"
            })
        
        if exclusions:
            questions.append({
                "question": f"Quelles sont les exclusions de l'assurance {insurance_type} ?",
                "ground_truth": ", ".join(exclusions[:5]) + "...",
                "source": f"assurances_final.jsonl - {doc.get('document_id', '')}"
            })
        
        if franchise:
            questions.append({
                "question": f"Quelle est la franchise de l'assurance {insurance_type} ?",
                "ground_truth": franchise,
                "source": f"assurances_final.jsonl - {doc.get('document_id', '')}"
            })
        
        # Specific coverage questions
        if "bris_de_glace" in covers:
            questions.append({
                "question": f"L'assurance {insurance_type} couvre-t-elle le bris de glace ?",
                "ground_truth": "Oui, le bris de glace est couvert.",
                "source": f"assurances_final.jsonl - {doc.get('document_id', '')}"
            })
        else:
            questions.append({
                "question": f"L'assurance {insurance_type} couvre-t-elle le bris de glace ?",
                "ground_truth": "Non, le bris de glace est exclu.",
                "source": f"assurances_final.jsonl - {doc.get('document_id', '')}"
            })
        
        if "vol" in covers:
            questions.append({
                "question": f"L'assurance {insurance_type} couvre-t-elle le vol du véhicule ?",
                "ground_truth": "Oui, le vol est couvert selon conditions.",
                "source": f"assurances_final.jsonl - {doc.get('document_id', '')}"
            })
        else:
            questions.append({
                "question": f"L'assurance {insurance_type} couvre-t-elle le vol du véhicule ?",
                "ground_truth": "Non, le vol est exclu.",
                "source": f"assurances_final.jsonl - {doc.get('document_id', '')}"
            })
    
    return questions[:count]

def extract_questions_from_faq(docs, count):
    """Extract questions from FAQ documents."""
    questions = []
    for doc in docs:
        for q in doc.get("questions", []):
            questions.append({
                "question": q.get("question", ""),
                "ground_truth": q.get("answer", ""),
                "source": f"faq_final.jsonl - {doc.get('document_id', '')}"
            })
    
    return questions[:count]

def extract_questions_from_vehicules(docs, count):
    """Extract questions from vehicules documents."""
    questions = []
    for doc in docs:
        vehicle = doc.get("vehicle", {})
        vehicle_id = doc.get("vehicle_id", "")
        brand = vehicle.get("brand", "")
        model = vehicle.get("model", "")
        category = vehicle.get("category", "")
        transmission = vehicle.get("transmission", "")
        seats = vehicle.get("seats", 0)
        luggage = vehicle.get("luggage_capacity", "")
        pricing = doc.get("pricing", "")
        features = vehicle.get("features", [])
        
        questions.append({
            "question": f"Quelle est la catégorie du {brand} {model} ({vehicle_id}) ?",
            "ground_truth": f"Catégorie {category}",
            "source": f"vehicules_final.jsonl - {doc.get('document_id', '')}"
        })
        
        questions.append({
            "question": f"Quelle transmission pour le {brand} {model} ?",
            "ground_truth": transmission,
            "source": f"vehicules_final.jsonl - {doc.get('document_id', '')}"
        })
        
        questions.append({
            "question": f"Combien de places pour le {brand} {model} ?",
            "ground_truth": f"{seats} places",
            "source": f"vehicules_final.jsonl - {doc.get('document_id', '')}"
        })
        
        questions.append({
            "question": f"Quel est le tarif journalier de la catégorie {category} ?",
            "ground_truth": pricing,
            "source": f"vehicules_final.jsonl - {doc.get('document_id', '')}"
        })
        
        if features:
            questions.append({
                "question": f"Quels équipements pour le {brand} {model} ?",
                "ground_truth": ", ".join(features),
                "source": f"vehicules_final.jsonl - {doc.get('document_id', '')}"
            })
        
        # Maintenance questions from vehicle
        maintenance = doc.get("maintenance_notes", "")
        if "vidange" in maintenance:
            import re
            match = re.search(r"vidange tous les (\d+) km", maintenance)
            if match:
                questions.append({
                    "question": f"À quelle fréquence la vidange pour le {brand} {model} ?",
                    "ground_truth": f"Tous les {match.group(1)} km",
                    "source": f"vehicules_final.jsonl - {doc.get('document_id', '')}"
                })
    
    return questions[:count]

def extract_questions_from_maintenance(docs, count):
    """Extract questions from maintenance documents."""
    questions = []
    for doc in docs:
        category = doc.get("vehicle_category", "")
        proc_type = doc.get("procedure_type", "")
        steps = doc.get("steps", [])
        diagnosis = doc.get("diagnosis", [])
        qa_gold = doc.get("qa_gold", [])
        
        for qa in qa_gold:
            questions.append({
                "question": qa.get("question", ""),
                "ground_truth": qa.get("answer", ""),
                "source": f"maintenance.jsonl - {doc.get('document_id', '')}"
            })
        
        if steps:
            questions.append({
                "question": f"Quelles sont les étapes de {proc_type} pour la catégorie {category} ?",
                "ground_truth": " | ".join(steps[:5]),
                "source": f"maintenance.jsonl - {doc.get('document_id', '')}"
            })
        
        if diagnosis:
            questions.append({
                "question": f"Quels symptômes pour {proc_type} catégorie {category} ?",
                "ground_truth": ", ".join(diagnosis[:5]),
                "source": f"maintenance.jsonl - {doc.get('document_id', '')}"
            })
    
    return questions[:count]

def main():
    print("Loading documents...")
    
    # Load all documents
    contrats_docs = load_jsonl(DATASET_DIR / "politiques_contrats.jsonl")
    assurances_docs = load_jsonl(DATASET_DIR / "assurances_final.jsonl")
    faq_docs = load_jsonl(DATASET_DIR / "faq_final.jsonl")
    vehicules_docs = load_jsonl(DATASET_DIR / "vehicules_final.jsonl")
    maintenance_docs = load_jsonl(DATASET_DIR / "maintenance.jsonl")
    
    all_questions = []
    
    # Extract questions per category
    print("Extracting questions from politiques_contrats...")
    contrats_q = extract_questions_from_politiques_contrats(contrats_docs, TARGET_COUNTS["contrats"])
    for i, q in enumerate(contrats_q):
        all_questions.append({
            "id": f"Q{len(all_questions)+1:03d}",
            "question": q["question"],
            "categorie": "contrats",
            "agent_attendu": "contrats",
            "ground_truth": q["ground_truth"],
            "source_attendue": q["source"]
        })
    
    print("Extracting questions from assurances...")
    assurances_q = extract_questions_from_assurances(assurances_docs, TARGET_COUNTS["assurances"])
    for i, q in enumerate(assurances_q):
        all_questions.append({
            "id": f"Q{len(all_questions)+1:03d}",
            "question": q["question"],
            "categorie": "assurances",
            "agent_attendu": "assurances",
            "ground_truth": q["ground_truth"],
            "source_attendue": q["source"]
        })
    
    print("Extracting questions from FAQ...")
    faq_q = extract_questions_from_faq(faq_docs, TARGET_COUNTS["faq"])
    for i, q in enumerate(faq_q):
        all_questions.append({
            "id": f"Q{len(all_questions)+1:03d}",
            "question": q["question"],
            "categorie": "faq",
            "agent_attendu": "faq",
            "ground_truth": q["ground_truth"],
            "source_attendue": q["source"]
        })
    
    print("Extracting questions from vehicules...")
    vehicules_q = extract_questions_from_vehicules(vehicules_docs, TARGET_COUNTS["vehicules"])
    for i, q in enumerate(vehicules_q):
        all_questions.append({
            "id": f"Q{len(all_questions)+1:03d}",
            "question": q["question"],
            "categorie": "vehicules",
            "agent_attendu": "vehicules",
            "ground_truth": q["ground_truth"],
            "source_attendue": q["source"]
        })
    
    print("Extracting questions from maintenance...")
    maintenance_q = extract_questions_from_maintenance(maintenance_docs, TARGET_COUNTS["maintenance"])
    for i, q in enumerate(maintenance_q):
        all_questions.append({
            "id": f"Q{len(all_questions)+1:03d}",
            "question": q["question"],
            "categorie": "maintenance",
            "agent_attendu": "maintenance",
            "ground_truth": q["ground_truth"],
            "source_attendue": q["source"]
        })
    
    # Add 5 multi-turn questions (with context)
    multi_turn = [
        {
            "id": f"Q{len(all_questions)+1:03d}",
            "question": "Quelle est la franchise de l'assurance CDW Premium ?",
            "categorie": "assurances",
            "agent_attendu": "assurances",
            "ground_truth": "La franchise de la CDW Premium varie de 1 000 à 2 500 MAD.",
            "source_attendue": "faq_final.jsonl / assurances_final.jsonl",
            "multi_turn_context": "Question précédente : 'Quelles sont les formules d'assurance disponibles ?'"
        },
        {
            "id": f"Q{len(all_questions)+1:03d}",
            "question": "Et pour la Full Protection ?",
            "categorie": "assurances",
            "agent_attendu": "assurances",
            "ground_truth": "La franchise de la Full Protection varie de 0 à 500 MAD.",
            "source_attendue": "faq_final.jsonl / assurances_final.jsonl",
            "multi_turn_context": "Question précédente : 'Quelle est la franchise de l'assurance CDW Premium ?'"
        },
        {
            "id": f"Q{len(all_questions)+1:03d}",
            "question": "Quel est le kilométrage inclus pour ce contrat ?",
            "categorie": "contrats",
            "agent_attendu": "contrats",
            "ground_truth": "1400 km/semaine inclus, 2 MAD par km supplémentaire",
            "source_attendue": "politiques_contrats.jsonl",
            "multi_turn_context": "Question précédente : 'Quelles sont les conditions du contrat RAB-2025-027 ?'"
        },
        {
            "id": f"Q{len(all_questions)+1:03d}",
            "question": "Et la caution ?",
            "categorie": "contrats",
            "agent_attendu": "contrats",
            "ground_truth": "5 000 MAD",
            "source_attendue": "politiques_contrats.jsonl",
            "multi_turn_context": "Question précédente : 'Quel est le kilométrage inclus pour ce contrat ?'"
        },
        {
            "id": f"Q{len(all_questions)+1:03d}",
            "question": "Quelle vidange pour cette catégorie ?",
            "categorie": "maintenance",
            "agent_attendu": "maintenance",
            "ground_truth": "Le seuil de vidange est de 12000 km.",
            "source_attendue": "maintenance.jsonl",
            "multi_turn_context": "Question précédente : 'Quels sont les seuils de maintenance pour la catégorie C ?'"
        },
    ]
    
    for q in multi_turn:
        all_questions.append(q)
    
    # Add 5 off-topic questions
    off_topic = [
        {
            "id": f"Q{len(all_questions)+1:03d}",
            "question": "Quelle heure est-il ?",
            "categorie": "hors_sujet",
            "agent_attendu": "hors_sujet",
            "ground_truth": "refus/hors périmètre",
            "source_attendue": "N/A"
        },
        {
            "id": f"Q{len(all_questions)+1:03d}",
            "question": "Quel temps fait-il à Paris ?",
            "categorie": "hors_sujet",
            "agent_attendu": "hors_sujet",
            "ground_truth": "refus/hors périmètre",
            "source_attendue": "N/A"
        },
        {
            "id": f"Q{len(all_questions)+1:03d}",
            "question": "Comment cuisiner des pâtes ?",
            "categorie": "hors_sujet",
            "agent_attendu": "hors_sujet",
            "ground_truth": "refus/hors périmètre",
            "source_attendue": "N/A"
        },
        {
            "id": f"Q{len(all_questions)+1:03d}",
            "question": "Qui a gagné la coupe du monde 2022 ?",
            "categorie": "hors_sujet",
            "agent_attendu": "hors_sujet",
            "ground_truth": "refus/hors périmètre",
            "source_attendue": "N/A"
        },
        {
            "id": f"Q{len(all_questions)+1:03d}",
            "question": "Quelle est la capitale de l'Australie ?",
            "categorie": "hors_sujet",
            "agent_attendu": "hors_sujet",
            "ground_truth": "refus/hors périmètre",
            "source_attendue": "N/A"
        },
    ]
    
    for q in off_topic:
        all_questions.append(q)
    
    # Add 3 questions with contract numbers
    contract_questions = [
        {
            "id": f"Q{len(all_questions)+1:03d}",
            "question": "Quelle est la franchise du contrat RAB-2025-027 ?",
            "categorie": "contrats",
            "agent_attendu": "contrats",
            "ground_truth": "Le contrat RAB-2025-027 inclut des assurances avec des franchises variables selon la formule choisie.",
            "source_attendue": "politiques_contrats.jsonl + graphe Neo4j",
            "has_contract_number": True
        },
        {
            "id": f"Q{len(all_questions)+1:03d}",
            "question": "Quels véhicules sont couverts par le contrat CASA-2024-014 ?",
            "categorie": "contrats",
            "agent_attendu": "contrats",
            "ground_truth": "Le contrat CASA-2024-014 couvre un véhicule de catégorie A (Économique) à l'agence Casablanca Gauthier.",
            "source_attendue": "politiques_contrats.jsonl + graphe Neo4j",
            "has_contract_number": True
        },
        {
            "id": f"Q{len(all_questions)+1:03d}",
            "question": "Quelle assurance est incluse dans le contrat MRK-2026-041 ?",
            "categorie": "contrats",
            "agent_attendu": "contrats",
            "ground_truth": "Le contrat MRK-2026-041 pour un touriste étranger en catégorie D SUV inclut les assurances selon le contrat.",
            "source_attendue": "politiques_contrats.jsonl + graphe Neo4j",
            "has_contract_number": True
        },
    ]
    
    for q in contract_questions:
        all_questions.append(q)
    
    # Write dataset
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for q in all_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    
    print(f"\nDataset created: {OUTPUT_FILE}")
    print(f"Total questions: {len(all_questions)}")
    
    # Count by category
    from collections import Counter
    cat_counts = Counter(q["categorie"] for q in all_questions)
    print("Per category:", dict(cat_counts))
    
    # Show samples
    print("\n--- Sample questions ---")
    for q in all_questions[:10]:
        print(f"{q['id']} [{q['categorie']}]: {q['question'][:80]}...")

if __name__ == "__main__":
    main()