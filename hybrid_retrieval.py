#!/usr/bin/env python3
"""
Module de retrieval hybride : BM25 + Vectoriel (ChromaDB) + RRF + Cross-encoder reranking.
"""

import json
import logging
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from rank_bm25 import BM25Okapi
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch

LOG_FORMAT = "%(asctime)s [HYBRID] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("hybrid_retrieval")

# Chemins
DATA_DIR = Path("/home/tawil/Bureau/stage d'ete/preprocessing/output")
CHROMA_DIR = Path("/home/tawil/Bureau/stage d'ete/preprocessing/output/chroma_db")
BM25_INDEX_DIR = Path("/home/tawil/Bureau/stage d'ete/preprocessing/output/bm25_indexes")

FILES = {
    "contrats": "chunks_contrats.jsonl",
    "assurances": "chunks_assurances.jsonl",
    "faq": "chunks_faq.jsonl",
    "maintenance": "chunks_maintenance.jsonl",
    "vehicules": "chunks_vehicules.jsonl",
}

SOURCE_FILES = {
    "contrats": "politiques_contrats.jsonl",
    "assurances": "assurances_final.jsonl",
    "faq": "faq_final.jsonl",
    "vehicules": "vehicules_final.jsonl",
    "maintenance": "maintenance.jsonl",
}

# Modèles
_embedding_model = None
_cross_encoder = None
_chroma_client = None
_bm25_indexes = {}


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma_client


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info("Chargement BGE-M3...")
        device = "cpu"  # CPU pour éviter OOM GPU
        logger.info(f"Device utilisé : {device}")
        _embedding_model = SentenceTransformer("BAAI/bge-m3", device=device)
        logger.info("Embedding OK")
    return _embedding_model


def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        try:
            logger.info("Chargement cross-encoder ms-marco-MiniLM-L6-v2...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device=device)
            logger.info("Cross-encoder OK")
        except Exception as e:
            logger.warning(f"Impossible de charger le cross-encoder: {e}. Reranking désactivé.")
            _cross_encoder = "ERROR"
    return _cross_encoder


def preprocess_text(text: str) -> List[str]:
    """Prétraitement simple pour le français : minuscules, tokenisation basique."""
    text = text.lower()
    # Garder lettres, chiffres, et certains caractères utiles
    tokens = re.findall(r"[a-zàâäéèêëïîôöùûüç0-9]+", text)
    return tokens


def build_bm25_index(category: str) -> Optional[BM25Okapi]:
    """Construit l'index BM25 pour une catégorie à partir du JSONL."""
    filename = FILES[category]
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        logger.error(f"Fichier introuvable: {filepath}")
        return None
    
    logger.info(f"Construction index BM25 pour {category}...")
    
    corpus = []
    documents = []
    metadatas = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            text = chunk.get("text", "")
            tokens = preprocess_text(text)
            corpus.append(tokens)
            documents.append(text)
            metadata = chunk.get("metadata", {})
            metadata["agent"] = category
            metadatas.append(metadata)
    
    if not corpus:
        logger.warning(f"Aucun chunk pour {category}")
        return None
    
    bm25 = BM25Okapi(corpus)
    logger.info(f"Index BM25 {category} construit: {len(corpus)} documents")
    
    return bm25


def save_bm25_index(category: str, bm25: BM25Okapi, documents: List[str], metadatas: List[Dict]):
    """Sauvegarde l'index BM25 et les données associées."""
    BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path = BM25_INDEX_DIR / f"bm25_{category}.pkl"
    
    data = {
        "bm25": bm25,
        "documents": documents,
        "metadatas": metadatas,
    }
    
    with open(index_path, "wb") as f:
        pickle.dump(data, f)
    
    logger.info(f"Index BM25 {category} sauvegardé: {index_path}")


def load_bm25_index(category: str) -> Optional[Tuple[BM25Okapi, List[str], List[Dict]]]:
    """Charge l'index BM25 sauvegardé."""
    index_path = BM25_INDEX_DIR / f"bm25_{category}.pkl"
    
    if not index_path.exists():
        logger.warning(f"Index BM25 non trouvé pour {category}: {index_path}")
        return None
    
    try:
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Index BM25 {category} chargé: {len(data['documents'])} documents")
        return data["bm25"], data["documents"], data["metadatas"]
    except Exception as e:
        logger.error(f"Erreur chargement index BM25 {category}: {e}")
        return None


def build_all_bm25_indexes():
    """Construit et sauvegarde tous les index BM25."""
    logger.info("=== CONSTRUCTION DE TOUS LES INDEX BM25 ===")
    
    for category in FILES.keys():
        try:
            bm25 = build_bm25_index(category)
            if bm25:
                # Recharger les docs pour sauvegarder
                filename = FILES[category]
                filepath = DATA_DIR / filename
                documents = []
                metadatas = []
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        chunk = json.loads(line)
                        documents.append(chunk.get("text", ""))
                        metadata = chunk.get("metadata", {})
                        metadata["agent"] = category
                        metadatas.append(metadata)
                
                save_bm25_index(category, bm25, documents, metadatas)
        except Exception as e:
            logger.error(f"Erreur construction index {category}: {e}")
    
    logger.info("=== INDEX BM25 TERMINÉS ===")


def search_vector(question: str, category: str, top_k: int = 20) -> List[Dict]:
    """Recherche vectorielle dans ChromaDB."""
    try:
        client = get_chroma_client()
        collection = client.get_collection(category)
        
        embedding_model = get_embedding_model()
        query_embedding = embedding_model.encode(
            question, normalize_embeddings=True, batch_size=8, show_progress_bar=False
        ).tolist()
        
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        docs = result["documents"][0] if result["documents"] else []
        metas = result["metadatas"][0] if result["metadatas"] else []
        distances = result["distances"][0] if result["distances"] else []
        
        results = []
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
            results.append({
                "text": doc,
                "metadata": meta,
                "score": 1.0 - dist,  # Similarité (plus haut = mieux)
                "rank": i + 1,
                "source": "vector"
            })
        
        return results
    except Exception as e:
        logger.error(f"Erreur recherche vectorielle {category}: {e}")
        return []


def search_bm25(question: str, category: str, top_k: int = 20) -> List[Dict]:
    """Recherche BM25 sur l'index."""
    # Essayer de charger l'index
    loaded = load_bm25_index(category)
    if loaded is None:
        # Essayer de le construire à la volée
        logger.warning(f"Index BM25 {category} absent, construction à la volée...")
        bm25 = build_bm25_index(category)
        if bm25 is None:
            return []
        # Sauvegarder pour la prochaine fois
        filename = FILES[category]
        filepath = DATA_DIR / filename
        documents = []
        metadatas = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                documents.append(chunk.get("text", ""))
                metadata = chunk.get("metadata", {})
                metadata["agent"] = category
                metadatas.append(metadata)
        save_bm25_index(category, bm25, documents, metadatas)
    else:
        bm25, documents, metadatas = loaded
    
    tokens = preprocess_text(question)
    scores = bm25.get_scores(tokens)
    
    # Top k indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    results = []
    for rank, idx in enumerate(top_indices):
        if scores[idx] > 0:
            results.append({
                "text": documents[idx],
                "metadata": metadatas[idx],
                "score": scores[idx],
                "rank": rank + 1,
                "source": "bm25"
            })
    
    return results


def rrf_fusion(vector_results: List[Dict], bm25_results: List[Dict], k: int = 60) -> List[Dict]:
    """Fusion Reciprocal Rank Fusion des résultats vectoriel et BM25."""
    # Créer un dictionnaire unique par chunk (basé sur le texte + metadata source)
    all_results = {}
    
    for res in vector_results:
        key = (res["text"][:200], res["metadata"].get("source", ""))
        if key not in all_results or (res["rank"] is not None and (all_results[key]["vector_rank"] is None or res["rank"] < all_results[key]["vector_rank"])):
            all_results[key] = {
                "text": res["text"],
                "metadata": res["metadata"],
                "vector_rank": res["rank"],
                "bm25_rank": None,
                "vector_score": res["score"],
                "bm25_score": None,
            }
    
    for res in bm25_results:
        key = (res["text"][:200], res["metadata"].get("source", ""))
        if key not in all_results:
            all_results[key] = {
                "text": res["text"],
                "metadata": res["metadata"],
                "vector_rank": None,
                "bm25_rank": res["rank"],
                "vector_score": None,
                "bm25_score": res["score"],
            }
        elif res["rank"] is not None and (all_results[key]["bm25_rank"] is None or res["rank"] < all_results[key]["bm25_rank"]):
            all_results[key]["bm25_rank"] = res["rank"]
            all_results[key]["bm25_score"] = res["score"]
    
    # Calculer score RRF
    for item in all_results.values():
        rrf_score = 0.0
        if item["vector_rank"] is not None:
            rrf_score += 1.0 / (k + item["vector_rank"])
        if item["bm25_rank"] is not None:
            rrf_score += 1.0 / (k + item["bm25_rank"])
        item["rrf_score"] = rrf_score
    
    # Trier par score RRF décroissant
    sorted_results = sorted(all_results.values(), key=lambda x: x["rrf_score"], reverse=True)
    
    return sorted_results


def rerank_cross_encoder(question: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
    """Reranking avec cross-encoder."""
    cross_encoder = get_cross_encoder()
    
    if cross_encoder == "ERROR" or cross_encoder is None:
        logger.warning("Cross-encoder non disponible, retour sans reranking")
        return candidates[:top_k]
    
    if not candidates:
        return []
    
    try:
        # Préparer les paires (question, document)
        pairs = [(question, cand["text"]) for cand in candidates]
        
        # Scorer
        scores = cross_encoder.predict(pairs, show_progress_bar=False)
        
        # Associer les scores
        for i, cand in enumerate(candidates):
            cand["rerank_score"] = float(scores[i])
        
        # Trier par score rerank
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        
        logger.info(f"Reranking effectué sur {len(candidates)} candidats")
        return reranked[:top_k]
    except Exception as e:
        logger.error(f"Erreur reranking: {e}. Retour sans reranking.")
        return candidates[:top_k]


def retrieve_hybrid(question: str, category: str, top_k_final: int = 5) -> Tuple[List[str], List[Dict]]:
    """
    Retrieval hybride : Vectoriel + BM25 -> RRF -> Cross-encoder reranking.
    
    Returns:
        Tuple (documents, metadatas) compatible avec l'ancien format
    """
    logger.info(f"Retrieval hybride: '{question}' -> {category}")
    
    # 1. Recherche vectorielle (top 20)
    vector_results = search_vector(question, category, top_k=20)
    logger.debug(f"Vectoriel: {len(vector_results)} résultats")
    
    # 2. Recherche BM25 (top 20)
    bm25_results = search_bm25(question, category, top_k=20)
    logger.debug(f"BM25: {len(bm25_results)} résultats")
    
    # 3. Fusion RRF
    if not vector_results and not bm25_results:
        logger.warning("Aucun résultat des deux méthodes")
        return [], []
    
    fused = rrf_fusion(vector_results, bm25_results, k=60)
    logger.debug(f"Fusion RRF: {len(fused)} résultats uniques")
    
    # 4. Reranking cross-encoder (sur top ~20)
    reranked = rerank_cross_encoder(question, fused[:20], top_k=top_k_final)
    
    # 5. Format de sortie compatible
    documents = [r["text"] for r in reranked]
    metadatas = [r["metadata"] for r in reranked]
    
    logger.info(f"Retrieval hybride terminé: {len(documents)} chunks retournés")
    
    return documents, metadatas


if __name__ == "__main__":
    # Test rapide
    build_all_bm25_indexes()
    
    # Test retrieval
    test_questions = [
        ("Quelle est la pénalité de retard sur un contrat de location ?", "contrats"),
        ("Quelle est la franchise en cas d'accident ?", "assurances"),
        ("Quels documents pour louer une voiture ?", "faq"),
        ("Équipements Dacia Logan", "vehicules"),
        ("Prochaine vidange flotte", "maintenance"),
    ]
    
    for q, cat in test_questions:
        docs, metas = retrieve_hybrid(q, cat, top_k_final=5)
        print(f"\nQ: {q} [{cat}]")
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            source = meta.get("source", "?")
            print(f"  {i+1}. [{source}] {doc[:100]}...")