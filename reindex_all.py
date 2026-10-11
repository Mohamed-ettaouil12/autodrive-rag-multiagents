#!/usr/bin/env python3
"""
Script pour ajouter le champ 'source' aux chunks existants et ré-indexer ChromaDB.
"""

import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm


# Mapping agent -> nom du fichier source original
SOURCE_FILES = {
    "contrats": "politiques_contrats.jsonl",
    "assurances": "assurances_final.jsonl",
    "faq": "faq_final.jsonl",
    "vehicules": "vehicules_final.jsonl",
    "maintenance": "maintenance.jsonl",
}

OUTPUT_DIR = Path("/home/tawil/Bureau/stage d'ete/preprocessing/output")
CHROMA_DIR = Path("/home/tawil/Bureau/stage d'ete/preprocessing/output/chroma_db")

FILES = {
    "contrats": "chunks_contrats.jsonl",
    "assurances": "chunks_assurances.jsonl",
    "faq": "chunks_faq.jsonl",
    "maintenance": "chunks_maintenance.jsonl",
    "vehicules": "chunks_vehicules.jsonl",
}


def add_source_to_chunks():
    """Ajoute le champ 'source' à chaque chunk dans les fichiers JSONL."""
    print("=== AJOUT DU CHAMP 'SOURCE' AUX CHUNKS ===\n")

    for agent, filename in FILES.items():
        source_file = SOURCE_FILES[agent]
        input_path = OUTPUT_DIR / filename
        output_path = OUTPUT_DIR / filename

        print(f"Traitement: {agent} <- {source_file}")

        chunks = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                # Ajouter le champ source dans metadata
                if "metadata" not in chunk:
                    chunk["metadata"] = {}
                chunk["metadata"]["source"] = source_file
                chunks.append(chunk)

        # Réécrire le fichier
        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        print(f"  -> {len(chunks)} chunks mis à jour")


def create_embedding(text, embedding_model):
    vector = embedding_model.encode(
        text,
        normalize_embeddings=True,
        batch_size=8,
        show_progress_bar=False
    )
    return vector.tolist()


def reindex_chromadb():
    """Ré-indexe toutes les collections dans ChromaDB."""
    print("\n=== RÉ-INDEXATION CHROMADB ===\n")

    # Charger le modèle d'embedding
    print("Chargement BGE-M3...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device utilisé : {device}")
    embedding_model = SentenceTransformer("BAAI/bge-m3", device=device)
    print("Embedding OK\n")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    for agent, filename in FILES.items():
        print(f"--- {agent} ---")

        # Supprimer l'ancienne collection
        try:
            client.delete_collection(name=agent)
            print(f"  Collection '{agent}' supprimée")
        except Exception:
            pass

        # Créer nouvelle collection
        collection = client.get_or_create_collection(name=agent)

        # Charger les chunks
        chunks_path = OUTPUT_DIR / filename
        chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))

        print(f"  {len(chunks)} chunks à indexer")

        # Indexer par batch
        batch_size = 64
        for i in tqdm(range(0, len(chunks), batch_size), desc=f"  Indexation {agent}"):
            batch = chunks[i:i+batch_size]

            documents = []
            embeddings = []
            ids = []
            metadatas = []

            for item in batch:
                text = item["text"]
                documents.append(text)
                embeddings.append(create_embedding(text, embedding_model))
                ids.append(item.get("chunk_id", f"{agent}_{i}"))
                metadata = item.get("metadata", {})
                metadata["agent"] = agent
                metadatas.append(metadata)

            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )

        print(f"  -> Collection '{agent}': {collection.count()} chunks indexés")


def verify_sources():
    """Vérifie que les sources sont bien présentes."""
    print("\n=== VÉRIFICATION DES SOURCES ===\n")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    for agent in FILES.keys():
        collection = client.get_collection(agent)
        # Récupérer quelques échantillons
        result = collection.get(limit=3, include=["metadatas"])
        print(f"{agent}:")
        for meta in result["metadatas"]:
            source = meta.get("source", "MANQUANT")
            print(f"  - source: {source}")


if __name__ == "__main__":
    add_source_to_chunks()
    reindex_chromadb()
    verify_sources()
    print("\n=== TERMINÉ ===")