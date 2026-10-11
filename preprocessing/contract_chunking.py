#!/usr/bin/env python3
"""
chunking_metadonnees_politiques_contrats.py
=============================================

Pipeline de chunking et d'enrichissement des metadonnees pour le corpus
"Politiques & Contrats" (agent = "politiques_contrats") du PFE AutoDrive Maroc.

Reference : Cahier des Charges, section 4.2 - meme strategie que le module
"vehicules" (chunking_metadonnees_vehicules.py), adaptee au schema de ce corpus :
    - representation tabulaire complete pour les donnees structurees (entities)
    - chunking narratif ~512 tokens (chevauchement 50) pour le contenu des
      sections, en respectant le decoupage "sections" deja fourni par la source
    - un chunk = une paire question/reponse pour les FAQ (qa_gold), en
      conservant le lien evidence_section_ids (tracabilite / evaluation RAGAS)

ENTREE : JSONL, schema "politiques_contrats.jsonl" :
    document_id, agent, title, document_type, synthetic, version,
    effective_date, agency, client_profile, scenario_tags[], entities{},
    sections[{section_id, title, content}],
    qa_gold[{question, answer, evidence_section_ids[]}]

SORTIE : JSONL de chunks {chunk_id, chunk_type, chunk_index, text,
    token_count, metadata} - metadata plate (str/int/float/bool uniquement),
    directement compatible avec collection.add(metadatas=...) de ChromaDB.

USAGE :
    python chunking_metadonnees_politiques_contrats.py \
        --input politiques_contrats.jsonl \
        --output chunks_politiques_contrats.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# 1. Tokenizer (identique au module "vehicules" : tiktoken si accessible,
#    sinon repli par mots - ne fait jamais planter le pipeline).
# ---------------------------------------------------------------------------

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
    pass


def count_tokens(text: str) -> int:
    return len(_encode(text))


def split_narrative(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Fenetre glissante de max_tokens / chevauchement overlap_tokens.
    Texte tenant deja dans max_tokens -> renvoye tel quel (1 chunk)."""
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
# 2. Metadonnees
# ---------------------------------------------------------------------------

def join_list(values: Any) -> str:
    """Aplatit une liste en chaine 'a, b, c' - necessaire car les metadatas
    ChromaDB doivent rester str/int/float/bool (jamais de liste imbriquee)."""
    if not values:
        return ""
    if isinstance(values, str):
        return values
    return ", ".join(str(v) for v in values)


def base_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Metadonnees communes a tous les chunks issus d'un meme document."""
    return {
        "document_id": record.get("document_id", ""),
        "agent": record.get("agent", "politiques_contrats"),
        "title": record.get("title", ""),
        "document_type": record.get("document_type", ""),
        "agency": record.get("agency", ""),
        "client_profile": record.get("client_profile", ""),
        "scenario_tags": join_list(record.get("scenario_tags")),
        "synthetic": bool(record.get("synthetic", False)),
        # Ce schema fournit nativement version et effective_date (contrairement
        # au corpus "vehicules") : on les lit directement, pas de repli
        # arbitraire. Seul "author" reste absent de la source -> repli explicite.
        "version": record.get("version") or "1.0",
        "update_date": record.get("effective_date") or "",
        "author": record.get("author") or "AutoDrive Maroc - import automatique",
    }


def make_chunk_id(base_id: str, chunk_type: str, index: int) -> str:
    raw = f"{base_id}:{chunk_type}:{index}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 3. Constructeurs de chunks
# ---------------------------------------------------------------------------

def build_entities_chunk(record: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    """
    Chunk 'entities' : representation tabulaire complete des entites
    structurees (CDC 4.2). Un seul chunk, jamais fragmente - ce sont les
    memes entites qui alimentent le graphe Neo4j (CDC 5.3), donc chunk et
    graphe restent coherents sur les memes valeurs.
    """
    entities = record.get("entities") or {}
    lignes = [
        f"Titre: {record.get('title', '')}",
        f"Type de document: {record.get('document_type', '')}",
    ]
    for k, v in entities.items():
        lignes.append(f"{k}: {v}")
    texte = "\n".join(lignes)
    n_tokens = count_tokens(texte)

    meta_entites = dict(meta)
    for k, v in entities.items():
        meta_entites[f"entity_{k}"] = v if isinstance(v, (str, int, float, bool)) else str(v)

    return {
        "chunk_id": make_chunk_id(record.get("document_id", ""), "entities", 0),
        "chunk_type": "entities",
        "chunk_index": 0,
        "text": texte,
        "token_count": n_tokens,
        "metadata": {**meta_entites, "source_field": "entities"},
    }


def build_section_chunks(
    record: dict[str, Any], meta: dict[str, Any], max_tokens: int, overlap_tokens: int
) -> list[dict[str, Any]]:
    """
    Un chunk par section source (CDC 4.2 : chunking narratif ~512 tokens).
    Le decoupage 'sections' fourni par la source est respecte comme unite
    semantique naturelle ; si le contenu d'une section depasse max_tokens,
    elle est elle-meme re-decoupee en fenetres avec chevauchement.
    """
    chunks = []
    for section in record.get("sections") or []:
        section_id = section.get("section_id", "")
        section_title = section.get("title", "")
        contenu = section.get("content", "") or ""
        texte_complet = f"{section_title}\n{contenu}".strip()
        morceaux = split_narrative(texte_complet, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

        for i, morceau in enumerate(morceaux):
            chunks.append({
                "chunk_id": make_chunk_id(section_id or record.get("document_id", ""), "section", i),
                "chunk_type": "section",
                "chunk_index": i,
                "text": morceau,
                "token_count": count_tokens(morceau),
                "metadata": {
                    **meta,
                    "source_field": "sections",
                    "section_id": section_id,
                    "section_title": section_title,
                    "chunk_total": len(morceaux),
                },
            })
    return chunks


def build_faq_chunks(record: dict[str, Any], meta: dict[str, Any], max_tokens: int) -> list[dict[str, Any]]:
    """
    Un chunk par paire Q/R (CDC 4.2), en conservant evidence_section_ids pour
    tracer la ou les sections qui justifient la reponse (utile pour
    l'evaluation RAGAS et pour l'audit).
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
                f"{max_tokens} tokens ({n_tokens}) - conservee entiere.",
                file=sys.stderr,
            )

        chunks.append({
            "chunk_id": make_chunk_id(record.get("document_id", ""), "faq", i),
            "chunk_type": "faq",
            "chunk_index": i,
            "text": texte,
            "token_count": n_tokens,
            "metadata": {
                **meta,
                "source_field": "qa_gold",
                "question": question,
                "evidence_section_ids": join_list(qa.get("evidence_section_ids")),
            },
        })
    return chunks


def chunk_record(record: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Applique les 3 constructeurs de chunks a un document politique/contrat."""
    meta = base_metadata(record)
    chunks: list[dict[str, Any]] = []
    chunks.append(build_entities_chunk(record, meta))
    chunks += build_section_chunks(record, meta, args.narrative_max_tokens, args.narrative_overlap)
    chunks += build_faq_chunks(record, meta, args.faq_max_tokens)
    return chunks


# ---------------------------------------------------------------------------
# 4. Lecture / ecriture JSONL
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
# 5. Programme principal
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunking + metadonnees pour le corpus JSONL de l'agent 'politiques_contrats' (AutoDrive Maroc)."
    )
    parser.add_argument("--input", default="politiques_contrats.jsonl", help="Fichier JSONL source")
    parser.add_argument("--output", default="chunks_politiques_contrats.jsonl", help="Fichier JSONL de sortie (chunks)")
    parser.add_argument("--stats-output", default=None, help="Fichier JSON de statistiques (defaut: <output>.stats.json)")
    parser.add_argument("--narrative-max-tokens", type=int, default=512)
    parser.add_argument("--narrative-overlap", type=int, default=50)
    parser.add_argument("--faq-max-tokens", type=int, default=256)
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

    print(f"[ok] {rapport['chunks_total']} chunks ecrits -> {output_path} (stats: {stats_path})", file=sys.stderr)


if __name__ == "__main__":
    main()
