

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# 1. Tokenizer
# ---------------------------------------------------------------------------
# On essaie tiktoken (meme famille de tokenizer que GPT-4o / text-embedding-
# 3-small, cf. section 6 du CDC) pour un compte de tokens exact. tiktoken doit
# telecharger son fichier d'encodage au premier appel : si la machine n'a pas
# acces a internet en sortie (cas courant en environnement d'entreprise, et
# coherent avec l'exigence CDC "donnees hebergees localement"), ou si le
# paquet n'est pas installe, on bascule automatiquement sur une approximation
# par mots, sans jamais faire planter le pipeline.

_TOKENIZER_NAME = "approximation par mots (tiktoken indisponible)"
_WORD_RE = re.compile(r"\S+")


def _encode_fallback(text: str) -> list[str]:
    return _WORD_RE.findall(text or "")


def _decode_fallback(tokens: list[str]) -> str:
    return " ".join(tokens)


_encode = _encode_fallback
_decode = _decode_fallback

try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")
    _enc.encode("test")  # force le chargement reel (declenche l'appel reseau ici)

    def _encode(text: str) -> list[int]:  # type: ignore[no-redef]
        return _enc.encode(text or "")

    def _decode(tokens: list[int]) -> str:  # type: ignore[no-redef]
        return _enc.decode(tokens)

    _TOKENIZER_NAME = "tiktoken (cl100k_base)"
except Exception:
    pass  # on garde le repli par mots defini plus haut


def count_tokens(text: str) -> int:
    return len(_encode(text))


# ---------------------------------------------------------------------------
# 2. Decoupage (chunking) d'un texte narratif long
# ---------------------------------------------------------------------------

def split_narrative(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """
    Fenetre glissante de max_tokens avec chevauchement overlap_tokens entre
    fenetres consecutives (evite de couper une idee sans contexte).
    Si le texte tient deja dans max_tokens, il est renvoye tel quel (1 seul chunk).
    Chaine vide/None -> aucun chunk.
    """
    text = (text or "").strip()
    if not text:
        return []

    tokens = _encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    chunks: list[str] = []
    step = max(1, max_tokens - overlap_tokens)
    start = 0
    while start < len(tokens):
        window = tokens[start:start + max_tokens]
        chunks.append(_decode(window).strip())
        if start + max_tokens >= len(tokens):
            break
        start += step
    return chunks


# ---------------------------------------------------------------------------
# 3. Metadonnees
# ---------------------------------------------------------------------------

def base_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """
    Metadonnees communes heritees du document source, appliquees a chaque
    chunk issu de ce document. Dict volontairement PLAT (pas de listes ni de
    sous-dicts) pour rester directement compatible avec les metadatas de
    ChromaDB.
    """
    vehicle = record.get("vehicle") or {}
    return {
        "document_id": record.get("document_id", ""),
        "vehicle_id": record.get("vehicle_id", ""),
        "agent": record.get("agent", "vehicules"),
        "title": record.get("title", ""),
        "agency": record.get("agency", ""),
        "vehicle_category": record.get("vehicle_category", ""),
        "brand": vehicle.get("brand", ""),
        "model": vehicle.get("model", ""),
        "synthetic": bool(record.get("synthetic", False)),
        # Le CDC (4.2) demande categorie / date de mise a jour / auteur / version.
        # Ces 3 derniers champs sont ABSENTS du JSON source fourni : on applique
        # un repli explicite plutot que de les omettre silencieusement. A
        # remplacer par les vraies metadonnees documentaires des que le corpus
        # complet (phase 2, section 4.1 du CDC) les fournit.
        "update_date": record.get("update_date") or date.today().isoformat(),
        "author": record.get("author") or "AutoDrive Maroc - import automatique",
        "version": record.get("version") or "lot_initial_v1",
    }


def make_chunk_id(document_id: str, chunk_type: str, index: int) -> str:
    raw = f"{document_id}:{chunk_type}:{index}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 4. Constructeurs de chunks par type
# ---------------------------------------------------------------------------

def build_fiche_technique_chunk(record: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    """
    Chunk 'fiche_technique' : representation tabulaire COMPLETE du vehicule
    (CDC 4.2 : "representation tabulaire complete pour ... les fiches
    vehicules"). Un seul chunk par vehicule, jamais fragmente.
    """
    v = record.get("vehicle") or {}
    lignes = [
        f"Titre: {record.get('title', '')}",
        f"Agence: {record.get('agency', '')}",
        f"Categorie vehicule: {record.get('vehicle_category', '')}",
        f"Marque: {v.get('brand', '')}",
        f"Modele: {v.get('model', '')}",
        f"Annee: {v.get('year') or 'non renseignee'}",
        f"Transmission: {v.get('transmission', '')}",
        f"Carburant: {v.get('fuel', '')}",
        f"Places: {v.get('seats', '')}",
        f"Capacite coffre: {v.get('luggage_capacity', '')}",
        f"Moteur: {v.get('engine', '')}",
        f"Puissance: {v.get('power', '')}",
        f"Consommation moyenne: {v.get('average_consumption', '')}",
        f"Kilometrage actuel: {v.get('current_mileage', '')}",
        f"Tarification: {record.get('pricing', '')}",
        f"Equipements: {', '.join(record.get('features') or [])}",
        f"Scenarios d'usage: {', '.join(record.get('usage_scenarios') or [])}",
        f"Profils clients adaptes: {', '.join(record.get('client_profile') or [])}",
        f"Conditions conducteur: {record.get('driver_requirements', '')}",
    ]
    texte = "\n".join(lignes)
    n_tokens = count_tokens(texte)

    if n_tokens > 1024:
        print(
            f"[avertissement] fiche technique {record.get('document_id')} "
            f"depasse 1024 tokens ({n_tokens}) et n'est pourtant pas fragmentee "
            f"(choix delibere CDC 4.2) - a surveiller si le corpus complet "
            f"apporte des fiches plus longues.",
            file=sys.stderr,
        )

    return {
        "chunk_id": make_chunk_id(record.get("document_id", ""), "fiche_technique", 0),
        "chunk_type": "fiche_technique",
        "chunk_index": 0,
        "text": texte,
        "token_count": n_tokens,
        "metadata": {**meta, "source_field": "vehicle+pricing+features+usage"},
    }


def build_narrative_chunks(
    record: dict[str, Any],
    meta: dict[str, Any],
    field_name: str,
    chunk_type: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[dict[str, Any]]:
    """
    Chunks pour un champ de texte narratif long (maintenance_notes,
    insurance_notes). Fenetre glissante de max_tokens / chevauchement
    overlap_tokens (CDC 4.2 : "chunks de 512 tokens pour les textes narratifs").
    """
    texte_source = record.get(field_name, "") or ""
    morceaux = split_narrative(texte_source, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

    chunks = []
    for i, morceau in enumerate(morceaux):
        chunks.append({
            "chunk_id": make_chunk_id(record.get("document_id", ""), chunk_type, i),
            "chunk_type": chunk_type,
            "chunk_index": i,
            "text": morceau,
            "token_count": count_tokens(morceau),
            "metadata": {**meta, "source_field": field_name, "chunk_total": len(morceaux)},
        })
    return chunks


def build_faq_chunks(record: dict[str, Any], meta: dict[str, Any], max_tokens: int) -> list[dict[str, Any]]:
    """
    Un chunk par paire Question/Reponse (CDC 4.2 : "parsing structure
    question/reponse pour conserver la paire Q&A comme unite semantique").
    La paire n'est jamais coupee, meme si elle depasse max_tokens (rare sur
    ce corpus : simple avertissement dans ce cas).
    """
    chunks = []
    for i, qa in enumerate(record.get("qa_gold") or []):
        question = (qa.get("question") or "").strip()
        reponse = (qa.get("answer") or "").strip()
        texte = f"Q: {question}\nR: {reponse}"
        n_tokens = count_tokens(texte)

        if n_tokens > max_tokens:
            print(
                f"[avertissement] FAQ {record.get('document_id')}#{i} depasse "
                f"{max_tokens} tokens ({n_tokens}) - conservee entiere (la paire "
                f"Q/R n'est jamais fragmentee).",
                file=sys.stderr,
            )

        chunks.append({
            "chunk_id": make_chunk_id(record.get("document_id", ""), "faq", i),
            "chunk_type": "faq",
            "chunk_index": i,
            "text": texte,
            "token_count": n_tokens,
            "metadata": {**meta, "source_field": "qa_gold", "question": question},
        })
    return chunks


def chunk_record(record: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Applique les 4 constructeurs de chunks a un document vehicule."""
    meta = base_metadata(record)
    chunks: list[dict[str, Any]] = []
    chunks.append(build_fiche_technique_chunk(record, meta))
    chunks += build_narrative_chunks(
        record, meta, "maintenance_notes", "maintenance",
        args.narrative_max_tokens, args.narrative_overlap,
    )
    chunks += build_narrative_chunks(
        record, meta, "insurance_notes", "assurance",
        args.narrative_max_tokens, args.narrative_overlap,
    )
    chunks += build_faq_chunks(record, meta, args.faq_max_tokens)
    return chunks


# ---------------------------------------------------------------------------
# 5. Lecture / ecriture JSONL
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for numero_ligne, ligne in enumerate(f, start=1):
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                yield json.loads(ligne)
            except json.JSONDecodeError as e:
                print(f"[erreur] ligne {numero_ligne} ignoree (JSON invalide) : {e}", file=sys.stderr)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 6. Programme principal
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunking + metadonnees pour le corpus JSONL de l'agent 'vehicules' (AutoDrive Maroc)."
    )
    parser.add_argument("--input", default="vehicules_final.jsonl", help="Fichier JSONL source")
    parser.add_argument("--output", default="chunks_vehicules.jsonl", help="Fichier JSONL de sortie (chunks)")
    parser.add_argument("--stats-output", default=None, help="Fichier JSON de statistiques (defaut: <output>.stats.json)")
    parser.add_argument("--narrative-max-tokens", type=int, default=512, help="Taille max (tokens) des chunks narratifs")
    parser.add_argument("--narrative-overlap", type=int, default=50, help="Chevauchement (tokens) entre chunks narratifs")
    parser.add_argument("--faq-max-tokens", type=int, default=256, help="Plafond (tokens) attendu pour un chunk FAQ")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    stats_path = Path(args.stats_output) if args.stats_output else output_path.with_suffix(".stats.json")

    if not input_path.exists():
        print(f"[erreur] fichier introuvable : {input_path}", file=sys.stderr)
        sys.exit(1)

    total_documents = 0
    all_chunks: list[dict[str, Any]] = []
    compte_par_type: Counter[str] = Counter()
    tokens_par_type: Counter[str] = Counter()

    for record in read_jsonl(input_path):
        total_documents += 1
        chunks = chunk_record(record, args)
        for c in chunks:
            compte_par_type[c["chunk_type"]] += 1
            tokens_par_type[c["chunk_type"]] += c["token_count"]
        all_chunks.extend(chunks)

    write_jsonl(output_path, all_chunks)

    rapport = {
        "tokenizer": _TOKENIZER_NAME,
        "documents_source": total_documents,
        "chunks_total": sum(compte_par_type.values()),
        "par_type": {
            t: {
                "chunks": compte_par_type[t],
                "tokens_moyen": round(tokens_par_type[t] / compte_par_type[t], 1) if compte_par_type[t] else 0,
                "tokens_total": tokens_par_type[t],
            }
            for t in compte_par_type
        },
        "fichier_sortie": str(output_path),
    }
    stats_path.write_text(json.dumps(rapport, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Rapport de chunking ===", file=sys.stderr)
    print(f"Tokenizer         : {_TOKENIZER_NAME}", file=sys.stderr)
    print(f"Documents source  : {total_documents}", file=sys.stderr)
    print(f"Chunks generes    : {rapport['chunks_total']}", file=sys.stderr)
    for t, d in rapport["par_type"].items():
        print(f"  - {t:<16} {d['chunks']:>4} chunks | {d['tokens_moyen']:>6} tokens en moyenne", file=sys.stderr)
    print(f"Sortie chunks     : {output_path}", file=sys.stderr)
    print(f"Sortie stats      : {stats_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
