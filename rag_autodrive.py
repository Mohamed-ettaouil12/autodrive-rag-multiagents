from pathlib import Path
import json
from tqdm import tqdm

import chromadb
from sentence_transformers import SentenceTransformer
import ollama


# =====================================================
# PATHS
# =====================================================

DATA_DIR = Path(
    "/home/tawil/Bureau/stage d'ete/preprocessing/output/"
)

CHROMA_DIR = Path(
    "/home/tawil/Bureau/stage d'ete/preprocessing/output/chroma_db/"
)


FILES = {

    "assurances":
        "chunks_assurances.jsonl",

    "contrats":
        "chunks_contrats.jsonl",

    "faq":
        "chunks_faq.jsonl",

    "maintenance":
        "chunks_maintenance.jsonl",

    "vehicules":
        "chunks_vehicules.jsonl"
}



# =====================================================
# EMBEDDING MODEL
# =====================================================

print("Chargement BGE-M3...")

import torch

device = "cpu"

print("Device utilisé :", device)

embedding_model = SentenceTransformer(
    "BAAI/bge-m3",
    device=device
)

print("Embedding OK")



def create_embedding(text):

    vector = embedding_model.encode(
    text,
    normalize_embeddings=True,
    batch_size=8,
    show_progress_bar=False
    )
    return vector.tolist()



# =====================================================
# LOAD JSONL
# =====================================================


def load_jsonl(file):

    path = DATA_DIR / file

    data=[]

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            data.append(
                json.loads(line)
            )


    return data



# =====================================================
# CREATE CHROMADB
# =====================================================


client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)



def create_database():


    print("\nCréation ChromaDB...\n")


    for agent,file in FILES.items():


        print(
            f"Traitement : {agent}"
        )


        chunks = load_jsonl(file)



        collection = client.get_or_create_collection(
            name=agent
        )


        # éviter duplication

        if collection.count()>0:

            print(
                "Collection déjà remplie :",
                collection.count()
            )

            continue



        batch_size=64



        for i in tqdm(
            range(
                0,
                len(chunks),
                batch_size
            )
        ):


            batch = chunks[
                i:i+batch_size
            ]


            documents=[]
            embeddings=[]
            ids=[]
            metadatas=[]



            for index,item in enumerate(batch):


                text=item["text"]


                documents.append(text)


                embeddings.append(
                    create_embedding(text)
                )


                ids.append(
                    item.get(
                        "chunk_id",
                        f"{agent}_{i+index}"
                    )
                )


                metadata=item.get(
                    "metadata",
                    {}
                )


                metadata["agent"]=agent


                metadatas.append(
                    metadata
                )



            collection.add(

                ids=ids,

                documents=documents,

                embeddings=embeddings,

                metadatas=metadatas
            )



        print(
            "Nombre chunks :",
            collection.count()
        )



# =====================================================
# RETRIEVAL
# =====================================================


def search_database(
        question,
        agent="assurances",
        top_k=5
):


    collection = client.get_collection(
        agent
    )


    query_embedding = create_embedding(
        question
    )


    result = collection.query(

        query_embeddings=[
            query_embedding
        ],

        n_results=top_k
    )



    documents=result["documents"][0]

    metadata=result["metadatas"][0]


    return documents,metadata




# =====================================================
# LLM OLLAMA
# =====================================================


def generate_answer(
        question,
        documents
):


    context=""


    for doc in documents:

        context += "\n\n"+doc



    prompt=f"""

Tu es un assistant IA interne AutoDrive Maroc.

Répond uniquement avec le contexte fourni.

Si l'information n'existe pas dans le contexte,
dis clairement :
"Information non trouvée dans la base documentaire."


CONTEXTE:

{context}


QUESTION:

{question}


REPONSE:
"""



    response = ollama.chat(

        model="mistral",

        messages=[

            {

            "role":"user",

            "content":prompt

            }

        ]

    )



    return response["message"]["content"]




# =====================================================
# MAIN CHAT
# =====================================================


if __name__=="__main__":


    # Première exécution seulement
    create_database()



    print("\n====================")
    print(" AutoDrive RAG")
    print("====================")


    while True:


        question=input(
            "\nQuestion : "
        )


        if question=="exit":
            break



        docs,meta = search_database(
            question,
            agent="assurances"
        )


        answer=generate_answer(
            question,
            docs
        )


        print("\n--- REPONSE ---")

        print(answer)



        print("\n--- SOURCES ---")

        for m in meta:

            print(m)