#!/usr/bin/env python3
"""
chunking_metadonnees_faq.py
=============================================

Pipeline de chunking et d'enrichissement des metadonnees pour le corpus
"FAQ" (agent = "faq") du PFE AutoDrive Maroc.

Reference : Cahier des Charges, section 4.2 - meme famille que les modules
"vehicules" et "politiques_contrats" (tokenizer, structure JSONL, metadata
plate). Schema plus simple ici : pas de texte narratif ni d'entites
structurees, uniquement des paires question/reponse groupees par categorie.
Chaque paire Q/R = un chunk (unite semantique jamais fragmentee, CDC 4.2).

ENTREE : JSONL, schema "faq_final.jsonl" :
    document_id, agent, category, synthetic,
    questions[{question, answer, related_agent, difficulty, keywords[]}]

SORTIE : JSONL de chunks {chunk_id, chunk_type, chunk_index, text,
    token_count, metadata} - metadata plate (str/int/float/bool uniquement),
    directement compatible avec collection.add(metadatas=...) de ChromaDB.

Le champ related_agent (routage vers l'agent LangGraph pertinent - vehicules,
politiques_contrats, assurances, maintenance, faq) et difficulty (simple /
complexe / realiste) sont conserves en metadata : utiles pour le routage
Superviseur (CDC 5.4) et pour stratifier l'evaluation RAGAS par difficulte.

USAGE :
    python chunking_metadonnees_faq.py \
        --input faq_final.jsonl \
        --output chunks_faq.jsonl
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
# 1. Tokenizer (identique aux modules "vehicules" / "politiques_contrats" :
#    tiktoken si accessible, sinon repli par mots - ne plante jamais).
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
    _enc.encode("test")

    def _encode(text: str) -> list[int]:  # type: ignore[no-redef]
        return _enc.encode(text or "")

    def _decode(tokens: list[int]) -> str:  # type: ignore[no-redef]
        return _enc.decode(tokens)

    _TOKENIZER_NAME = "tiktoken (cl100k_base)"
except Exception:
    pass


def count_tokens(text: str) -> int:
    return len(_encode(text))


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
    """Metadonnees communes a tous les chunks issus d'un meme document FAQ."""
    return {
        "document_id": record.get("document_id", ""),
        "agent": record.get("agent", "faq"),
        "category": record.get("category", ""),
        "synthetic": bool(record.get("synthetic", False)),
        # Ni version ni date ne sont presentes dans ce schema (contrairement a
        # politiques_contrats) : repli explicite, comme pour le corpus vehicules.
        "version": record.get("version") or "lot_initial_v1",
        "update_date": record.get("update_date") or "",
        "author": record.get("author") or "AutoDrive Maroc - import automatique",
    }


def make_chunk_id(document_id: str, chunk_type: str, index: int) -> str:
    raw = f"{document_id}:{chunk_type}:{index}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 3. Constructeur de chunks
# ---------------------------------------------------------------------------

def build_faq_chunks(record: dict[str, Any], meta: dict[str, Any], max_tokens: int) -> list[dict[str, Any]]:
    """
    Un chunk par paire Q/R de 'questions' (CDC 4.2 : la paire Q/R reste une
    unite semantique, jamais fragmentee). related_agent / difficulty /
    keywords sont conserves en metadata pour le routage et l'evaluation.
    """
    chunks = []
    for i, qa in enumerate(record.get("questions") or []):
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
                "source_field": "questions",
                "question": question,
                "related_agent": qa.get("related_agent", ""),
                "difficulty": qa.get("difficulty", ""),
                "keywords": join_list(qa.get("keywords")),
            },
        })
    return chunks


def chunk_record(record: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    meta = base_metadata(record)
    return build_faq_chunks(record, meta, args.faq_max_tokens)


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
        description="Chunking + metadonnees pour le corpus JSONL de l'agent 'faq' (AutoDrive Maroc)."
    )
    parser.add_argument("--input", default="faq_final.jsonl", help="Fichier JSONL source")
    parser.add_argument("--output", default="chunks_faq.jsonl", help="Fichier JSONL de sortie (chunks)")
    parser.add_argument("--stats-output", default=None, help="Fichier JSON de statistiques (defaut: <output>.stats.json)")
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