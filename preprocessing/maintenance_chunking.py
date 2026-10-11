#!/usr/bin/env python3
"""
chunking_metadonnees_maintenance.py
====================================

Pipeline Chunking + Metadata enrichment
pour l'agent maintenance du projet AutoDrive Maroc.

ENTREE:
    maintenance.jsonl

SORTIE:
    chunks_maintenance.jsonl

Compatible:
    - Embeddings
    - ChromaDB / Qdrant
    - BM25
    - Retrieval hybride RAG
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from datetime import date
from typing import Any, Iterator
from collections import Counter


# ============================================================
# 1. TOKENIZER
# ============================================================

_WORD_RE = re.compile(r"\S+")


def encode(text):

    return _WORD_RE.findall(text or "")



def decode(tokens):

    return " ".join(tokens)



def count_tokens(text):

    return len(encode(text))


# ============================================================
# 2. CHUNKING TEXTE LONG
# ============================================================

def split_text(text, max_tokens=512, overlap=50):

    text = (text or "").strip()

    if not text:
        return []


    tokens = encode(text)


    if len(tokens) <= max_tokens:
        return [text]


    chunks=[]

    step=max_tokens-overlap

    start=0


    while start < len(tokens):

        part=tokens[start:start+max_tokens]

        chunks.append(
            decode(part)
        )


        start += step


    return chunks



# ============================================================
# 3. METADATA
# ============================================================

def create_metadata(record):


    return {

        "document_id":
            record.get("document_id",""),


        "agent":
            record.get("agent","maintenance"),


        "domain":
            "maintenance",


        "vehicle_id":
            record.get("vehicle_id",""),


        "vehicle_category":
            record.get("vehicle_category",""),


        "maintenance_type":
            record.get("maintenance_type",""),


        "priority":
            record.get("priority",""),


        "author":
            record.get(
                "author",
                "AutoDrive Maroc"
            ),


        "update_date":
            record.get(
                "update_date",
                date.today().isoformat()
            ),


        "version":
            record.get(
                "version",
                "v1"
            ),


        "synthetic":
            bool(
                record.get(
                    "synthetic",
                    False
                )
            )

    }



# ============================================================
# 4. CHUNK ID
# ============================================================

def generate_id(document_id, chunk_type,index):


    raw=f"{document_id}:{chunk_type}:{index}"


    return hashlib.sha1(
        raw.encode()
    ).hexdigest()[:16]



# ============================================================
# 5. PROCEDURE MAINTENANCE
# ============================================================

def build_procedure_chunk(record,metadata):


    steps=record.get(
        "steps",
        []
    )


    text=""


    text += (
        f"Type maintenance : "
        f"{record.get('maintenance_type','')}\n\n"
    )


    text += "Procedure:\n"


    for step in steps:

        text += f"- {step}\n"



    return {

        "chunk_id":
            generate_id(
                record["document_id"],
                "maintenance",
                0
            ),


        "chunk_type":
            "maintenance_procedure",


        "chunk_index":
            0,


        "text":
            text,


        "token_count":
            count_tokens(text),


        "metadata":
            {
                **metadata,
                "source_field":"steps"
            }

    }



# ============================================================
# 6. NOTES MAINTENANCE LONGUES
# ============================================================

def build_notes_chunks(
        record,
        metadata,
        max_tokens=512,
        overlap=50):


    notes=record.get(
        "maintenance_notes",
        ""
    )


    parts=split_text(
        notes,
        max_tokens,
        overlap
    )


    chunks=[]


    for i,p in enumerate(parts):


        chunks.append({

            "chunk_id":
                generate_id(
                    record["document_id"],
                    "maintenance_notes",
                    i
                ),


            "chunk_type":
                "maintenance_notes",


            "chunk_index":
                i,


            "text":
                p,


            "token_count":
                count_tokens(p),


            "metadata":
                {
                    **metadata,
                    "source_field":
                        "maintenance_notes"
                }

        })


    return chunks



# ============================================================
# 7. FAQ
# ============================================================

def build_faq_chunks(record,metadata):


    chunks=[]


    faq_list=record.get(
        "qa_gold",
        []
    )


    for i,item in enumerate(faq_list):


        text=(

            "Question : "
            + item.get("question","")
            +
            "\nRéponse : "
            +
            item.get("answer","")

        )


        chunks.append({

            "chunk_id":
                generate_id(
                    record["document_id"],
                    "faq",
                    i
                ),


            "chunk_type":
                "faq",


            "chunk_index":
                i,


            "text":
                text,


            "token_count":
                count_tokens(text),


            "metadata":
                {
                    **metadata,
                    "source_field":
                        "qa_gold"
                }

        })


    return chunks



# ============================================================
# 8. PROCESS DOCUMENT
# ============================================================

def process_record(record,args):


    metadata=create_metadata(record)


    chunks=[]


    chunks.append(
        build_procedure_chunk(
            record,
            metadata
        )
    )


    chunks.extend(
        build_notes_chunks(
            record,
            metadata,
            args.max_tokens,
            args.overlap
        )
    )


    chunks.extend(
        build_faq_chunks(
            record,
            metadata
        )
    )


    return chunks



# ============================================================
# 9. JSONL IO
# ============================================================

def read_jsonl(path):


    with open(
        path,
        encoding="utf-8"
    ) as f:


        for line in f:

            if line.strip():

                yield json.loads(line)



def write_jsonl(path,data):


    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:


        for item in data:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                )
                +"\n"
            )



# ============================================================
# 10. MAIN
# ============================================================

def main():


    parser=argparse.ArgumentParser()


    parser.add_argument(
        "--input",
        default="maintenance.jsonl"
    )


    parser.add_argument(
        "--output",
        default="chunks_maintenance.jsonl"
    )


    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512
    )


    parser.add_argument(
        "--overlap",
        type=int,
        default=50
    )


    args=parser.parse_args()



    all_chunks=[]

    stats=Counter()



    for record in read_jsonl(args.input):


        chunks=process_record(
            record,
            args
        )


        for c in chunks:

            stats[c["chunk_type"]]+=1


        all_chunks.extend(chunks)



    write_jsonl(
        args.output,
        all_chunks
    )


    print("\n===== RESULTAT =====")

    print(
        "Total chunks:",
        len(all_chunks)
    )


    for k,v in stats.items():

        print(
            k,
            ":",
            v
        )


    print(
        "Fichier:",
        args.output
    )



if __name__=="__main__":

    main()