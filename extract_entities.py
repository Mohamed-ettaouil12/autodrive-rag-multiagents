import argparse
import json
import logging
import re
import time
from pathlib import Path

import ollama
from tqdm import tqdm

LOG_FORMAT = "%(asctime)s [GRAPH] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("extract_entities")

CATEGORIES = {
    "politiques_contrats": "dataset/politiques_contrats.jsonl",
    "assurances": "dataset/assurances_final.jsonl",
    "faq": "dataset/faq_final.jsonl",
    "maintenance": "dataset/maintenance.jsonl",
    "vehicules": "dataset/vehicules_final.jsonl",
}

OUTPUT_FILE = "extraction_resultats.jsonl"
ERROR_FILE = "erreurs_parsing.jsonl"
MODEL = "mistral"
CONFIDENCE_THRESHOLD = 0.85

FEW_SHOT_EXAMPLES = """Exemple:
Texte: "Contrat CASA-2024-014, véhicule ADM-A-CASA-01-001 Dacia Sandero cat A. Assurance CDW Premium franchise 1500 MAD exclut vol."
JSON: {"entites":[{"type":"Contrat","nom":"CASA-2024-014","proprietes":{}},{"type":"Vehicule","nom":"ADM-A-CASA-01-001","proprietes":{"marque":"Dacia","modele":"Sandero","categorie":"A"}},{"type":"Assurance","nom":"CDW Premium","proprietes":{"franchise_mad":1500}},{"type":"Sinistre","nom":"vol","proprietes":{"type":"exclusion"}}],"relations":[{"source":"CASA-2024-014","type":"COUVRE","cible":"ADM-A-CASA-01-001","confiance":0.95},{"source":"CASA-2024-014","type":"INCLUT","cible":"CDW Premium","confiance":0.9},{"source":"CDW Premium","type":"EXCLUT","cible":"vol","confiance":0.98}]}"""

EXTRACTION_PROMPT = """Extrais UNIQUEMENT ces entités métier d'AutoDrive Maroc:
- Contrat: numéro (ex: CASA-2024-014, RAB-2025-027, MRK-2026-041)
- Vehicule: immatriculation ADM-... (ex: ADM-A-CASA-01-001), marque, modèle, catégorie A-F
- Assurance: nom produit (Basic, CDW Standard, CDW Premium, Theft Protection, Full Protection), code, franchise MAD
- Client: type (particulier_marocain, entreprise_marocaine, touriste_etranger)
- Politique: titre politique (ex: "Politique d'annulation")
- Categorie: A/B/C/D/E/F (Économique, Compacte, Berline, SUV, Monospace, Utilitaire)
- Maintenance: type (vidange, pneus, freinage, batterie, climatisation), seuil km/mois
- Sinistre: type exclusion (vol, bris_de_glace, etc.)

Relations: Contrat-COUVRE->Vehicule, Contrat-INCLUT->Assurance, Vehicule-APPARTIENT_A->Categorie, Vehicule-A_HISTORIQUE->Maintenance, Assurance-EXCLUT->Sinistre, Client-A_SIGNE->Contrat.

{examples}

Règles: JSON seul. Confiance 0-1. Si aucune entité métier: {{"entites":[],"relations":[]}}
Ne crée JAMAIS d'entité dont le nom est vide, inconnu ou non mentionné explicitement dans le texte — ignore-la.

Texte: {text}

JSON:"""


def _parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end+1]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raw = raw.replace("'", '"')
        raw = re.sub(r',\s*}', '}', raw)
        raw = re.sub(r',\s*]', ']', raw)
        open_braces = raw.count("{")
        close_braces = raw.count("}")
        if open_braces > close_braces:
            raw += "}" * (open_braces - close_braces)
        open_brackets = raw.count("[")
        close_brackets = raw.count("]")
        if open_brackets > close_brackets:
            raw += "]" * (open_brackets - close_brackets)
        return json.loads(raw)


def extract_entities_from_chunk(chunk_text: str) -> dict:
    prompt = EXTRACTION_PROMPT.format(examples=FEW_SHOT_EXAMPLES, text=chunk_text[:1500])

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_ctx": 4096, "num_predict": 384},
        )
        raw_content = response["message"]["content"]
        parsed = json.loads(raw_content)

        entites = parsed.get("entites", [])
        relations = parsed.get("relations", [])

        for rel in relations:
            if rel.get("confiance", 1.0) < CONFIDENCE_THRESHOLD:
                rel["a_valider"] = True

        return {"entites": entites, "relations": relations, "raw_response": raw_content, "parse_error": None}

    except json.JSONDecodeError as e:
        logger.warning("Erreur parsing JSON (format=json): %s", e)
    except Exception as e:
        logger.error("Erreur appel LLM: %s", e)

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_ctx": 4096},
        )
        raw_content = response["message"]["content"]
        parsed = _parse_json_response(raw_content)

        entites = parsed.get("entites", [])
        relations = parsed.get("relations", [])

        for rel in relations:
            if rel.get("confiance", 1.0) < CONFIDENCE_THRESHOLD:
                rel["a_valider"] = True

        return {"entites": entites, "relations": relations, "raw_response": raw_content, "parse_error": None}

    except Exception as e:
        logger.warning("Échec complet parsing: %s", e)
        with open(ERROR_FILE, "a", encoding="utf-8") as ef:
            ef.write(json.dumps({"prompt": prompt, "raw_response": raw_content, "error": str(e)}, ensure_ascii=False) + "\n")
        return {"entites": [], "relations": [], "raw_response": raw_content if 'raw_content' in locals() else "", "parse_error": str(e)}


def load_chunks(category: str, max_chunks: int, offset: int = 0, source_depuis: str = None, source_jusqua: str = None) -> list:
    filepath = CATEGORIES.get(category)
    if not filepath or not Path(filepath).exists():
        logger.error("Fichier introuvable pour la catégorie: %s", category)
        return []

    chunks = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if i >= offset + max_chunks:
                break
            try:
                data = json.loads(line.strip())
                text = data.get("text") or data.get("content") or ""
                if not text and "sections" in data:
                    text = " ".join(s.get("content", "") for s in data["sections"])
                if not text and "vehicle" in data:
                    v = data["vehicle"]
                    text = f"{data.get('title', '')} {v.get('brand', '')} {v.get('model', '')} {data.get('pricing', '')} {data.get('maintenance_notes', '')} {data.get('insurance_notes', '')}"
                if not text and "entities" in data:
                    e = data["entities"]
                    text = f"{data.get('title', '')} Contrat {e.get('contract_number', '')} {e.get('agency_name', '')} {e.get('vehicle_category_name', '')} {data.get('late_return_rule', '')} {data.get('fuel_rule', '')}"
                if not text and "questions" in data:
                    text = " ".join(q.get("question", "") + " " + q.get("answer", "") for q in data["questions"])
                if not text and "steps" in data:
                    text = " ".join(data["steps"]) + " " + " ".join(data.get("diagnosis", [])) + " " + " ".join(data.get("actions", []))

                source_id = data.get("document_id", f"chunk_{i}")

                if source_depuis and source_id < source_depuis:
                    continue
                if source_jusqua and source_id > source_jusqua:
                    continue

                if text.strip():
                    chunks.append({
                        "category": category,
                        "source": source_id,
                        "text": text[:1500],
                        "metadata": data
                    })
            except json.JSONDecodeError:
                continue
    return chunks


def load_processed_sources(output_file: str) -> set:
    """Charge les sources déjà traitées depuis le fichier de sortie existant."""
    processed = set()
    if not Path(output_file).exists():
        return processed
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    source = data.get("source")
                    if source:
                        processed.add(source)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning("Impossible de lire le fichier de sortie existant: %s", e)
    return processed


def main():
    parser = argparse.ArgumentParser(description="Extraction entités/relations par LLM (reprenable)")
    parser.add_argument("--categorie", choices=list(CATEGORIES.keys()) + ["toutes"], default="toutes")
    parser.add_argument("--max-chunks", type=int, default=200)
    parser.add_argument("--offset", type=int, default=0, help="Décalage de départ dans les chunks (défaut: 0)")
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--resume", action="store_true", default=True, help="Reprendre depuis le fichier de sortie existant (défaut: activé)")
    parser.add_argument("--fresh", action="store_true", help="Ignorer le fichier existant et repartir de zéro")
    parser.add_argument("--source-depuis", type=str, help="ID source de départ (ex: ASS-0100)")
    parser.add_argument("--source-jusqua", type=str, help="ID source de fin (ex: ASS-0200)")
    args = parser.parse_args()

    if args.fresh:
        args.resume = False

    categories_to_process = list(CATEGORIES.keys()) if args.categorie == "toutes" else [args.categorie]

    all_chunks = []
    for cat in categories_to_process:
        chunks = load_chunks(cat, args.max_chunks, args.offset, args.source_depuis, args.source_jusqua)
        all_chunks.extend(chunks)
        logger.info("Catégorie %s: %d chunks chargés (offset=%d, max=%d)", cat, len(chunks), args.offset, args.max_chunks)

    if args.resume:
        processed_sources = load_processed_sources(args.output)
        if processed_sources:
            original_count = len(all_chunks)
            all_chunks = [c for c in all_chunks if c["source"] not in processed_sources]
            skipped = original_count - len(all_chunks)
            logger.info("Mode reprise: %d chunks déjà traités ignorés, %d restants", skipped, len(all_chunks))
        else:
            logger.info("Aucun chunk traité précédemment trouvé, démarrage frais")

    total_chunks = len(all_chunks)
    logger.info("Total: %d chunks à traiter (modèle: %s)", total_chunks, MODEL)

    if total_chunks == 0:
        logger.info("Rien à traiter.")
        return

    Path(ERROR_FILE).touch()

    processed = 0
    success_count = 0
    total_time = 0.0
    start_time = time.time()

    with open(args.output, "a", encoding="utf-8") as out_f:
        for i, chunk in enumerate(all_chunks):
            chunk_start = time.time()
            result = extract_entities_from_chunk(chunk["text"])
            elapsed = time.time() - chunk_start
            total_time += elapsed

            if result.get("parse_error") is None:
                success_count += 1

            output_line = {
                "categorie": chunk["category"],
                "source": chunk["source"],
                "entites": result["entites"],
                "relations": result["relations"],
                "confiance_moyenne": sum(r.get("confiance", 1.0) for r in result["relations"]) / len(result["relations"]) if result["relations"] else 1.0,
                "a_valider": any(r.get("a_valider", False) for r in result["relations"]),
                "temps_sec": round(elapsed, 1),
                "parse_ok": result.get("parse_error") is None,
            }

            out_f.write(json.dumps(output_line, ensure_ascii=False) + "\n")
            out_f.flush()
            processed += 1

            if processed % 10 == 0 or processed == total_chunks:
                avg_time = total_time / processed
                remaining = total_chunks - processed
                eta = avg_time * remaining
                elapsed_total = time.time() - start_time
                logger.info("Chunk %d/%d traité, temps moyen %.1fs, ETA %.0fs (%.1fmin), total écoulé %.0fs",
                           processed, total_chunks, avg_time, eta, eta/60, elapsed_total)

    avg_time = total_time / processed if processed else 0
    success_rate = success_count / processed * 100 if processed else 0
    total_elapsed = time.time() - start_time

    logger.info("=== STATISTIQUES ===")
    logger.info("Chunks traités: %d", processed)
    logger.info("Parsing réussi: %d/%d (%.1f%%)", success_count, processed, success_rate)
    logger.info("Temps moyen: %.1fs/chunk", avg_time)
    logger.info("Temps total: %.0fs (%.1fmin)", total_elapsed, total_elapsed/60)
    logger.info("Résultats dans: %s", args.output)
    logger.info("Erreurs parsing dans: %s", ERROR_FILE)

    with open(args.output, "r", encoding="utf-8") as f:
        lines = f.readlines()
        logger.info("=== EXEMPLES D'EXTRACTIONS (premiers 5) ===")
        for i, line in enumerate(lines[:5]):
            data = json.loads(line)
            logger.info("Exemple %d: cat=%s, source=%s, entites=%d, relations=%d, a_valider=%s, temps=%.1fs, parse_ok=%s",
                       i+1, data["categorie"], data["source"], len(data["entites"]), len(data["relations"]), data["a_valider"], data["temps_sec"], data["parse_ok"])
            for e in data["entites"][:3]:
                nom = e.get("nom") or e.get("name") or "(sans nom)"
                logger.info("  Entité: %s (%s) - %s", nom, e["type"], e.get("proprietes", {}))
            for r in data["relations"][:3]:
                logger.info("  Relation: %s -[%s]-> %s (confiance=%.2f, a_valider=%s)",
                           r["source"], r["type"], r["cible"], r.get("confiance", 1.0), r.get("a_valider", False))


if __name__ == "__main__":
    main()