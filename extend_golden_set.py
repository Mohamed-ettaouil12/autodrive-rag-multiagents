#!/usr/bin/env python3
"""
Extend the golden set dataset_eval.jsonl with:
- passages_attendus: list of chunk_ids that should be retrieved
- type_question: simple | multi-hop | sans_reponse | ambigue | chiffree
Target: ~80 questions (10 multi-hop, 8 sans_reponse, 5 ambigue, 10 chiffree)
"""

import json
from pathlib import Path

INPUT_FILE = Path("/home/tawil/Bureau/stage d'ete/dataset_eval.jsonl")
OUTPUT_FILE = Path("/home/tawil/Bureau/stage d'ete/dataset_eval_extended.jsonl")

# Mapping from existing questions to expected chunk_ids (based on chunk exploration)
PASSAGES_MAP = {
    # Contrats - PC-0001 (CASA-2024-014)
    "Q001": ["453580b8587299a1", "b982b80eab5ac1fa"],  # pièces requises
    "Q002": ["a9a55ce422a65375", "5dcddc609cf8e486"],  # tarif journalier
    "Q003": ["d0132c84df60de18", "e1ccce47506a29f2"],  # règle retard
    "Q004": ["6b447699cc37d841"],  # âge minimum
    "Q005": ["aad3a41ce9027ee3", "5dcddc609cf8e486"],  # caution
    "Q006": ["5dcddc609cf8e486", "aad3a41ce9027ee3"],  # calcul caution
    "Q007": ["96dd43c3435f122c", "e0b8ad599ff8c9e1"],  # kilométrage
    "Q008": ["e1ccce47506a29f2"],  # règle carburant
    # Contrats - PC-0002 (RAB-2025-027)
    "Q009": ["339dca84f67e95c2"],  # pièces requises
    "Q010": ["bb3dc2740d1fe145"],  # tarif journalier
    "Q011": ["06adf3e2e25869fa"],  # règle retard (similar to PC-0001)
    "Q012": ["339dca84f67e95c2"],  # âge minimum
    "Q013": ["bb3dc2740d1fe145"],  # caution
    "Q014": ["55220fa4f0afb0c8"],  # kilométrage illimité
    # Contrats - PC-0003 (MRK-2026-041)
    "Q015": ["80fa96c40a19802e"],  # pièces requises (entities chunk)
    # Assurances - ASS-0001, ASS-0002, ASS-0003
    "Q016": ["120353735f1c5b8d", "9fb1685e03f98f21"],  # couvertures Basic
    "Q017": ["5522c473eee4b1d6", "421ad7be0bb850bd"],  # exclusions Basic
    "Q018": ["6b3fca71874ef7d6", "a4028291e375f6cb"],  # franchise Basic
    "Q019": ["120353735f1c5b8d", "9fb1685e03f98f21"],  # bris de glace
    "Q020": ["120353735f1c5b8d", "9fb1685e03f98f21"],  # vol
    "Q021": ["120353735f1c5b8d", "9fb1685e03f98f21"],  # couvertures Basic (ASS-0002)
    "Q022": ["5522c473eee4b1d6", "421ad7be0bb850bd"],  # exclusions Basic (ASS-0002)
    "Q023": ["6b3fca71874ef7d6", "a4028291e375f6cb"],  # franchise Basic (ASS-0002)
    "Q024": ["120353735f1c5b8d", "9fb1685e03f98f21"],  # bris de glace (ASS-0002)
    "Q025": ["120353735f1c5b8d", "9fb1685e03f98f21"],  # vol (ASS-0002)
    "Q026": ["e35625c71add000b", "9fb1685e03f98f21"],  # couvertures CDW Standard
    "Q027": ["421ad7be0bb850bd", "5522c473eee4b1d6"],  # exclusions CDW Standard
    # FAQ
    "Q028": ["a6a8483be69da4d8"],  # agences disponibles
    "Q029": ["d622250d93c887cd"],  # horaires Casablanca
    "Q030": ["3d52e36ad2c99600"],  # horaires Rabat
    "Q031": ["ff82aafcaac60087"],  # horaires Marrakech
    "Q032": ["60c0e4966c1f8460"],  # catégories véhicules
    "Q033": ["377229599040a11d"],  # places monospace E
    "Q034": ["41c95745bc7da095"],  # conducteur 22 ans SUV D
    "Q035": ["13c7ef3874c9a9c2"],  # conducteur 23 ans monospace E
    "Q036": ["57164fafb8349d74"],  # personne 21 ans berline C
    "Q037": ["9222e86f68c4bc87"],  # entreprise conducteur désigné
    "Q038": ["d622250d93c887cd"],  # touriste étranger
    "Q039": ["60c0e4966c1f8460", "377229599040a11d"],  # voiture auto 5 personnes
    # Véhicules
    "Q040": ["c279493166dd23a1", "424753972306b0a0"],  # catégorie Dacia Sandero
    "Q041": ["2e825d01f62466b3", "424753972306b0a0"],  # transmission Dacia Sandero
    "Q042": ["37920a8e23eb6556", "424753972306b0a0"],  # places Dacia Sandero
    "Q043": ["ed7a40700f836ed0", "424753972306b0a0"],  # tarif catégorie A
    "Q044": ["605e6ad7dd6be54b", "424753972306b0a0"],  # vidange Dacia Sandero
    "Q045": ["c279493166dd23a1"],  # catégorie Renault Clio
    "Q046": ["2e825d01f62466b3"],  # transmission Renault Clio
    "Q047": ["37920a8e23eb6556"],  # places Renault Clio
    "Q048": ["ed7a40700f836ed0"],  # tarif catégorie A
    "Q049": ["605e6ad7dd6be54b"],  # vidange Renault Clio
    "Q050": ["c279493166dd23a1"],  # catégorie Hyundai i10
    # Maintenance
    "Q051": ["bf8a670f3575f23c", "69e815d8a151fa9f"],  # checkpoints avant location
    "Q052": ["2b7ec86a3a0979bf", "605e6ad7dd6be54b"],  # vidange catégorie A
    "Q053": ["c974fbdd02e3d73a"],  # seuils pneus freinage
    "Q054": ["674d4b24c0de15f2"],  # voyant rouge
    "Q055": ["fe5b1ca36f042689"],  # urgence critique
    "Q056": ["69e815d8a151fa9f"],  # étapes inspection
    "Q057": ["bf8a670f3575f23c", "69e815d8a151fa9f"],  # symptômes
    "Q058": ["d55eeab30a2788af", "e52bf539ccadd49f"],  # checkpoints retour
    "Q059": ["96d187070d5c5df2", "e52bf539ccadd49f"],  # seuil batterie B
    "Q060": ["2f4c0b06bb61d1cf", "e52bf539ccadd49f"],  # seuil climatisation B
    # Multi-turn (existing)
    "Q061": ["6b3fca71874ef7d6", "a4028291e375f6cb"],  # franchise CDW Premium
    "Q062": ["6b3fca71874ef7d6", "a4028291e375f6cb"],  # franchise Full Protection
    "Q063": ["55220fa4f0afb0c8", "bb3dc2740d1fe145"],  # kilométrage contrat
    "Q064": ["bb3dc2740d1fe145"],  # caution
    "Q065": ["2b7ec86a3a0979bf", "605e6ad7dd6be54b"],  # vidange catégorie
    # Hors sujet
    "Q066": [],  # heure
    "Q067": [],  # temps Paris
    "Q068": [],  # pâtes
    "Q069": [],  # coupe du monde
    "Q070": [],  # capitale Australie
    # Graph questions
    "Q071": ["f6bf617dc2369c1f", "5d744569e0dccb8c"],  # franchise contrat RAB
    "Q072": ["f6bf617dc2369c1f", "5d744569e0dccb8c"],  # véhicules contrat CASA
    "Q073": ["80fa96c40a19802e"],  # assurance contrat MRK
}

# Type classification for existing questions
TYPE_MAP = {
    # Simple factual questions
    "Q001": "simple", "Q002": "chiffree", "Q003": "simple", "Q004": "simple", "Q005": "chiffree",
    "Q006": "simple", "Q007": "chiffree", "Q008": "simple",
    "Q009": "simple", "Q010": "chiffree", "Q011": "simple", "Q012": "simple", "Q013": "chiffree",
    "Q014": "simple", "Q015": "simple",
    "Q016": "simple", "Q017": "simple", "Q018": "chiffree", "Q019": "simple", "Q020": "simple",
    "Q021": "simple", "Q022": "simple", "Q023": "chiffree", "Q024": "simple", "Q025": "simple",
    "Q026": "simple", "Q027": "simple",
    "Q028": "simple", "Q029": "simple", "Q030": "simple", "Q031": "simple", "Q032": "simple",
    "Q033": "simple", "Q034": "simple", "Q035": "simple", "Q036": "simple", "Q037": "simple",
    "Q038": "simple", "Q039": "simple",
    "Q040": "simple", "Q041": "simple", "Q042": "simple", "Q043": "chiffree", "Q044": "simple",
    "Q045": "simple", "Q046": "simple", "Q047": "simple", "Q048": "chiffree", "Q049": "simple",
    "Q050": "simple",
    "Q051": "simple", "Q052": "chiffree", "Q053": "chiffree", "Q054": "simple", "Q055": "simple",
    "Q056": "simple", "Q057": "simple", "Q058": "simple", "Q059": "chiffree", "Q060": "chiffree",
    # Multi-turn context questions
    "Q061": "multi-hop", "Q062": "multi-hop", "Q063": "multi-hop", "Q064": "multi-hop", "Q065": "multi-hop",
    # Hors sujet
    "Q066": "sans_reponse", "Q067": "sans_reponse", "Q068": "sans_reponse", "Q069": "sans_reponse", "Q070": "sans_reponse",
    # Graph questions
    "Q071": "multi-hop", "Q072": "multi-hop", "Q073": "multi-hop",
}

# New questions to add (to reach targets)
NEW_QUESTIONS = [
    # Multi-hop (need 10 total, have 8 already: Q061-Q065, Q071-Q073)
    {
        "id": "Q074",
        "question": "Quel est le tarif journalier et la caution pour le contrat CASA-2024-014 ?",
        "categorie": "contrats",
        "agent_attendu": "contrats",
        "ground_truth": "Tarif journalier : 220 MAD. Caution : 3000 MAD.",
        "source_attendue": "politiques_contrats.jsonl - PC-0001",
        "passages_attendus": ["5dcddc609cf8e486", "aad3a41ce9027ee3", "a9a55ce422a65375"],
        "type_question": "multi-hop"
    },
    {
        "id": "Q075",
        "question": "Quelles sont les franchises des assurances Basic et CDW Standard ?",
        "categorie": "assurances",
        "agent_attendu": "assurances",
        "ground_truth": "Basic : 8 000 à 15 000 MAD (ex: 10 000 ou 12 000 MAD). CDW Standard : franchise selon contrat.",
        "source_attendue": "assurances_final.jsonl - ASS-0001, ASS-0003",
        "passages_attendus": ["6b3fca71874ef7d6", "a4028291e375f6cb"],
        "type_question": "multi-hop"
    },
    {
        "id": "Q076",
        "question": "Un client avec le contrat RAB-2025-027 veut savoir la caution et le kilométrage inclus.",
        "categorie": "contrats",
        "agent_attendu": "contrats",
        "ground_truth": "Caution : 5000 MAD. Kilométrage : illimité.",
        "source_attendue": "politiques_contrats.jsonl - PC-0002",
        "passages_attendus": ["bb3dc2740d1fe145", "55220fa4f0afb0c8"],
        "type_question": "multi-hop"
    },

    # Sans réponse (need 8 total, have 5 already: Q066-Q070)
    {
        "id": "Q077",
        "question": "Quel est le prix de l'assurance pour un contrat inexistant XYZ-2024-999 ?",
        "categorie": "assurances",
        "agent_attendu": "assurances",
        "ground_truth": "Information non trouvée dans la base documentaire.",
        "source_attendue": "N/A",
        "passages_attendus": [],
        "type_question": "sans_reponse"
    },
    {
        "id": "Q078",
        "question": "Quelle est la procédure de maintenance pour un véhicule catégorie Z qui n'existe pas ?",
        "categorie": "maintenance",
        "agent_attendu": "maintenance",
        "ground_truth": "Information non trouvée dans la base documentaire.",
        "source_attendue": "N/A",
        "passages_attendus": [],
        "type_question": "sans_reponse"
    },
    {
        "id": "Q079",
        "question": "Combien coûte la location d'une Ferrari chez AutoDrive Maroc ?",
        "categorie": "vehicules",
        "agent_attendu": "vehicules",
        "ground_truth": "Information non trouvée dans la base documentaire.",
        "source_attendue": "N/A",
        "passages_attendus": [],
        "type_question": "sans_reponse"
    },

    # Ambiguës (need 5 total, have 0)
    {
        "id": "Q080",
        "question": "Combien ça coûte ?",
        "categorie": "contrats",
        "agent_attendu": "contrats",
        "ground_truth": "Le tarif dépend du contrat et de la catégorie de véhicule. Pour la catégorie A : 220 MAD/jour. Pour la catégorie C (contrat RAB-2025-027) : 8200 MAD/mois.",
        "source_attendue": "politiques_contrats.jsonl - PC-0001, PC-0002",
        "passages_attendus": ["5dcddc609cf8e486", "bb3dc2740d1fe145"],
        "type_question": "ambigue"
    },
    {
        "id": "Q081",
        "question": "Quelle est la franchise ?",
        "categorie": "assurances",
        "agent_attendu": "assurances",
        "ground_truth": "La franchise dépend de la formule d'assurance choisie. Basic : 8 000 à 15 000 MAD. CDW Premium : 1 000 à 2 500 MAD. Full Protection : 0 à 500 MAD.",
        "source_attendue": "assurances_final.jsonl - ASS-0001, ASS-0002, ASS-0003",
        "passages_attendus": ["6b3fca71874ef7d6", "a4028291e375f6cb"],
        "type_question": "ambigue"
    },
    {
        "id": "Q082",
        "question": "Quels sont les horaires ?",
        "categorie": "faq",
        "agent_attendu": "faq",
        "ground_truth": "Casablanca : 08:00-19:00 (lun-sam), 09:00-13:00 (dim). Rabat : 08:00-19:00 (lun-sam), 09:00-13:00 (dim). Marrakech : 08:00-20:00 (tous jours), 08:00-13:00 (jours fériés).",
        "source_attendue": "faq_final.jsonl - FAQ-0001",
        "passages_attendus": ["d622250d93c887cd", "3d52e36ad2c99600", "ff82aafcaac60087"],
        "type_question": "ambigue"
    },
    {
        "id": "Q083",
        "question": "Quelle vidange pour mon véhicule ?",
        "categorie": "maintenance",
        "agent_attendu": "maintenance",
        "ground_truth": "La fréquence de vidange dépend de la catégorie du véhicule. Catégorie A : tous les 10 000 km. Catégorie B : tous les 12 000 km. Catégorie C : selon préconisations constructeur.",
        "source_attendue": "maintenance.jsonl - MNT-0001, MNT-0002; vehicules_final.jsonl",
        "passages_attendus": ["605e6ad7dd6be54b", "2b7ec86a3a0979bf"],
        "type_question": "ambigue"
    },
    {
        "id": "Q084",
        "question": "Est-ce que l'assurance couvre ?",
        "categorie": "assurances",
        "agent_attendu": "assurances",
        "ground_truth": "Cela dépend de la formule d'assurance. Basic : responsabilité civile seulement (dommages aux tiers). CDW Standard : dommages collision partiel. CDW Premium : dommages collision complet + vol + bris de glace. Full Protection : tout risque avec franchise 0-500 MAD.",
        "source_attendue": "assurances_final.jsonl",
        "passages_attendus": ["120353735f1c5b8d", "9fb1685e03f98f21", "5522c473eee4b1d6"],
        "type_question": "ambigue"
    },

    # Chiffrées (need 10 total, have ~6 already: Q002, Q005, Q007, Q010, Q013, Q018, Q023, Q043, Q048, Q052, Q053, Q059, Q060)
    # Actually we have quite a few chiffree already. Let me add a few more specific ones.
    {
        "id": "Q085",
        "question": "Combien coûte le km supplémentaire pour le contrat CASA-2024-014 ?",
        "categorie": "contrats",
        "agent_attendu": "contrats",
        "ground_truth": "2,5 MAD par km supplémentaire.",
        "source_attendue": "politiques_contrats.jsonl - PC-0001",
        "passages_attendus": ["96dd43c3435f122c", "e0b8ad599ff8c9e1"],
        "type_question": "chiffree"
    },
    {
        "id": "Q086",
        "question": "Quel est le montant des frais de service carburant si le réservoir n'est pas plein à la restitution ?",
        "categorie": "contrats",
        "agent_attendu": "contrats",
        "ground_truth": "100 MAD de frais de service majorés du carburant manquant au tarif pompe.",
        "source_attendue": "politiques_contrats.jsonl - PC-0001",
        "passages_attendus": ["e1ccce47506a29f2"],
        "type_question": "chiffree"
    },
    {
        "id": "Q087",
        "question": "Quel est le pourcentage de pénalité par heure de retard après la tolérance d'une heure ?",
        "categorie": "contrats",
        "agent_attendu": "contrats",
        "ground_truth": "10% du tarif journalier par heure de retard.",
        "source_attendue": "politiques_contrats.jsonl - PC-0001",
        "passages_attendus": ["d0132c84df60de18", "e1ccce47506a29f2"],
        "type_question": "chiffree"
    },
    {
        "id": "Q088",
        "question": "Quelle est la franchise minimale pour l'assurance Basic ?",
        "categorie": "assurances",
        "agent_attendu": "assurances",
        "ground_truth": "8 000 MAD.",
        "source_attendue": "assurances_final.jsonl - ASS-0001",
        "passages_attendus": ["6b3fca71874ef7d6", "a4028291e375f6cb"],
        "type_question": "chiffree"
    },
    {
        "id": "Q089",
        "question": "Quel est le tarif mensuel de la catégorie C pour le contrat RAB-2025-027 ?",
        "categorie": "contrats",
        "agent_attendu": "contrats",
        "ground_truth": "8 200 MAD par mois.",
        "source_attendue": "politiques_contrats.jsonl - PC-0002",
        "passages_attendus": ["bb3dc2740d1fe145"],
        "type_question": "chiffree"
    },
    {
        "id": "Q090",
        "question": "À combien s'élève la caution pour un contrat entreprise catégorie C ?",
        "categorie": "contrats",
        "agent_attendu": "contrats",
        "ground_truth": "5 000 MAD.",
        "source_attendue": "politiques_contrats.jsonl - PC-0002",
        "passages_attendus": ["bb3dc2740d1fe145"],
        "type_question": "chiffree"
    },
    {
        "id": "Q091",
        "question": "Quel est le seuil de vidange pour la catégorie B ?",
        "categorie": "maintenance",
        "agent_attendu": "maintenance",
        "ground_truth": "12 000 km.",
        "source_attendue": "maintenance.jsonl - MNT-0002",
        "passages_attendus": ["2b7ec86a3a0979bf"],
        "type_question": "chiffree"
    },
    {
        "id": "Q092",
        "question": "Quel est le seuil de pneus pour la catégorie A ?",
        "categorie": "maintenance",
        "agent_attendu": "maintenance",
        "ground_truth": "40 000 km.",
        "source_attendue": "maintenance.jsonl - MNT-0001",
        "passages_attendus": ["c974fbdd02e3d73a"],
        "type_question": "chiffree"
    },
    {
        "id": "Q093",
        "question": "Quel est le seuil de freinage pour la catégorie A ?",
        "categorie": "maintenance",
        "agent_attendu": "maintenance",
        "ground_truth": "20 000 km.",
        "source_attendue": "maintenance.jsonl - MNT-0001",
        "passages_attendus": ["c974fbdd02e3d73a"],
        "type_question": "chiffree"
    },
]


def main():
    # Load existing dataset
    existing = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                existing.append(json.loads(line))

    # Fix duplicate IDs in existing (Q061 appears multiple times, Q066 appears multiple times)
    # Renumber them
    id_counter = 1
    id_mapping = {}
    for q in existing:
        old_id = q["id"]
        new_id = f"Q{id_counter:03d}"
        id_mapping[old_id] = new_id
        q["id"] = new_id
        id_counter += 1

    # Add passages_attendus and type_question to existing
    for q in existing:
        qid = q["id"]
        q["passages_attendus"] = PASSAGES_MAP.get(qid, [])
        q["type_question"] = TYPE_MAP.get(qid, "simple")

    # Add new questions with adjusted IDs
    for new_q in NEW_QUESTIONS:
        new_q["id"] = f"Q{id_counter:03d}"
        id_counter += 1
        existing.append(new_q)

    # Write extended dataset
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for q in existing:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"Extended dataset written to {OUTPUT_FILE}")
    print(f"Total questions: {len(existing)}")

    # Count by type
    from collections import Counter
    type_counts = Counter(q["type_question"] for q in existing)
    print("Type distribution:", dict(type_counts))

    # Count by category
    cat_counts = Counter(q["categorie"] for q in existing)
    print("Category distribution:", dict(cat_counts))

    # Verify passages_attendus exist
    all_chunk_ids = set()
    for cat in ["contrats", "assurances", "faq", "vehicules", "maintenance"]:
        filepath = Path(f"/home/tawil/Bureau/stage d'ete/preprocessing/output/chunks_{cat}.jsonl")
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        chunk = json.loads(line)
                        all_chunk_ids.add(chunk["chunk_id"])

    missing = []
    for q in existing:
        for pid in q.get("passages_attendus", []):
            if pid not in all_chunk_ids:
                missing.append((q["id"], pid))
    if missing:
        print(f"WARNING: {len(missing)} passage IDs not found in chunks:")
        for qid, pid in missing[:10]:
            print(f"  {qid}: {pid}")
    else:
        print("All passage IDs verified against chunk files.")


if __name__ == "__main__":
    main()