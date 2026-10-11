#!/usr/bin/env python3
"""
chunking_metadonnees_assurances_v2.py

Chunking + metadata enrichment
Agent : Assurances - AutoDrive Maroc
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date
from collections import Counter


# ==========================================================
# Tokenizer simple
# ==========================================================

_WORD_RE = re.compile(r"\S+")


def encode(text):
    return _WORD_RE.findall(text or "")


def decode(tokens):
    return " ".join(tokens)


def count_tokens(text):
    return len(encode(text))


# ==========================================================
# Chunking narratif
# ==========================================================

def split_text(text, max_tokens=512, overlap=50):

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



# ==========================================================
# Metadata
# ==========================================================

def create_metadata(record):


    return {

        "document_id":
            record.get("document_id",""),


        "agent":
            "assurances",


        "domain":
            "assurance",


        "insurance_type":
            record.get(
                "insurance_type",
                record.get("type","")
            ),


        "vehicle_category":
            record.get(
                "vehicle_category",
                ""
            ),


        "contract_id":
            record.get(
                "contract_id",
                ""
            ),


        "franchise":
            str(
                record.get(
                    "franchise",
                    ""
                )
            ),


        "version":
            record.get(
                "version",
                "v1"
            ),


        "update_date":
            record.get(
                "update_date",
                date.today().isoformat()
            ),


        "author":
            record.get(
                "author",
                "AutoDrive Maroc"
            )

    }



# ==========================================================
# Chunk ID
# ==========================================================

def make_id(doc_id,ctype,index):

    value=f"{doc_id}:{ctype}:{index}"

    return hashlib.sha1(
        value.encode()
    ).hexdigest()[:16]



# ==========================================================
# Recherche automatique des champs assurance
# ==========================================================

def get_field(record,names):


    for name in names:

        if name in record and record[name]:

            return record[name]


    return None



# ==========================================================
# Construction chunks assurance
# ==========================================================

def build_assurance_chunks(record,metadata):


    chunks=[]

    index=0


    sections={

        "garantie":[
            "garanties",
            "guarantees",
            "coverage",
            "couverture"
        ],


        "exclusion":[
            "exclusions",
            "exceptions"
        ],


        "franchise":[
            "franchise",
            "deductible"
        ],


        "sinistre":[
            "sinistre",
            "claims",
            "procedure_sinistre",
            "procedure"
        ],


        "regles":[
            "rules",
            "conditions",
            "notes"
        ]

    }



    for chunk_type,fields in sections.items():


        value=get_field(
            record,
            fields
        )


        if not value:
            continue



        if isinstance(value,list):

            text="\n".join(
                "- "+str(x)
                for x in value
            )

        else:

            text=str(value)



        parts=split_text(
            text,
            512,
            50
        )


        for part in parts:


            chunks.append({

                "chunk_id":
                    make_id(
                        record.get("document_id",""),
                        chunk_type,
                        index
                    ),


                "chunk_type":
                    "assurance_"+chunk_type,


                "chunk_index":
                    index,


                "text":
                    part,


                "token_count":
                    count_tokens(part),


                "metadata":
                    {
                        **metadata,
                        "source_field":
                            fields[0]
                    }

            })


            index +=1



    return chunks



# ==========================================================
# FAQ
# ==========================================================

def build_faq(record,metadata):


    chunks=[]


    faq=record.get(
        "qa_gold",
        record.get(
            "faq",
            []
        )
    )


    for i,item in enumerate(faq):


        if isinstance(item,dict):

            question=item.get(
                "question",
                ""
            )

            answer=item.get(
                "answer",
                ""
            )

        else:

            question=""
            answer=str(item)



        text=f"""
Question:
{question}

Réponse:
{answer}
"""


        chunks.append({

            "chunk_id":
                make_id(
                    record.get("document_id",""),
                    "faq",
                    i
                ),


            "chunk_type":
                "faq",


            "chunk_index":
                i,


            "text":
                text.strip(),


            "token_count":
                count_tokens(text),


            "metadata":
                {
                    **metadata,
                    "source_field":
                        "faq"
                }

        })


    return chunks



# ==========================================================
# Process document
# ==========================================================

def process(record):

    metadata=create_metadata(record)

    chunks=[]


    chunks.extend(
        build_assurance_chunks(
            record,
            metadata
        )
    )


    chunks.extend(
        build_faq(
            record,
            metadata
        )
    )


    return chunks



# ==========================================================
# JSONL
# ==========================================================

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



# ==========================================================
# Main
# ==========================================================

def main():

    parser=argparse.ArgumentParser()


    parser.add_argument(
        "--input",
        default="../dataset/assurances_final(4).jsonl"
    )


    parser.add_argument(
        "--output",
        default="output/chunks_assurances_v2.jsonl"
    )


    args=parser.parse_args()


    all_chunks=[]

    stats=Counter()


    for record in read_jsonl(args.input):


        chunks=process(record)


        for c in chunks:

            stats[c["chunk_type"]]+=1


        all_chunks.extend(chunks)



    write_jsonl(
        args.output,
        all_chunks
    )


    print("\n===== RESULTAT =====")

    print(
        "Nombre chunks:",
        len(all_chunks)
    )


    for k,v in stats.items():

        print(
            k,
            ":",
            v
        )


    print(
        "Output:",
        args.output
    )



if __name__=="__main__":

    main()