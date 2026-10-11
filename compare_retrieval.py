#!/usr/bin/env python3
"""
Script de comparaison : retrieval vectoriel seul (ancien) vs retrieval hybride (nouveau).
"""

import logging
from rag_autodrive import search_database
from hybrid_retrieval import retrieve_hybrid, search_vector

LOG_FORMAT = "%(asctime)s [COMPARE] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("compare")

# 10 questions métier : 2 par catégorie
TEST_QUESTIONS = [
    # Contrats
    ("Quelle est la pénalité de retard sur un contrat de location ?", "contrats"),
    ("Quels documents faut-il pour louer une voiture ?", "contrats"),
    
    # Assurances
    ("Quelle est la franchise en cas d'accident responsable ?", "assurances"),
    ("Quelles sont les exclusions de garantie ?", "assurances"),
    
    # FAQ
    ("Quels sont les horaires d'ouverture des agences ?", "faq"),
    ("Comment modifier une réservation ?", "faq"),
    
    # Véhicules
    ("Quels équipements possède la Dacia Logan ?", "vehicules"),
    ("Quelle est la consommation de la Dacia Sandero ?", "vehicules"),
    
    # Maintenance
    ("Quand faire la vidange d'un véhicule catégorie A ?", "maintenance"),
    ("Quelle est la procédure de contrôle avant location ?", "maintenance"),
]


def search_vector_only(question: str, category: str, top_k: int = 5):
    """Wrapper pour retrieval vectoriel seul (ancien système)."""
    docs, metas = search_database(question, agent=category, top_k=top_k)
    return docs, metas


def format_chunk(text: str, metadata: dict, max_len: int = 100) -> str:
    """Formate un chunk pour l'affichage."""
    source = metadata.get("source", "?")
    preview = text[:max_len].replace("\n", " ")
    return f"[{source}] {preview}"


def main():
    print("=" * 100)
    print("COMPARAISON RETRIEVAL : VECTORIEL SEUL (ANCIEN) vs HYBRIDE BM25+VECTORIEL+RRF+RERANK (NOUVEAU)")
    print("=" * 100)
    
    for i, (question, category) in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'='*100}")
        print(f"QUESTION {i}/10 [{category.upper()}] : {question}")
        print(f"{'='*100}")
        
        # Ancien système : vectoriel seul
        print("\n--- ANCIEN : VECTORIEL SEUL (ChromaDB) ---")
        try:
            vec_docs, vec_metas = search_vector_only(question, category, top_k=5)
            if not vec_docs:
                print("  Aucun résultat")
            else:
                for j, (doc, meta) in enumerate(zip(vec_docs, vec_metas)):
                    print(f"  {j+1}. {format_chunk(doc, meta)}")
        except Exception as e:
            print(f"  ERREUR: {e}")
        
        # Nouveau système : hybride
        print("\n--- NOUVEAU : HYBRIDE (BM25 + Vectoriel -> RRF -> Cross-encoder) ---")
        try:
            hyb_docs, hyb_metas = retrieve_hybrid(question, category, top_k_final=5)
            if not hyb_docs:
                print("  Aucun résultat")
            else:
                for j, (doc, meta) in enumerate(zip(hyb_docs, hyb_metas)):
                    print(f"  {j+1}. {format_chunk(doc, meta)}")
        except Exception as e:
            print(f"  ERREUR: {e}")
        
        print()


if __name__ == "__main__":
    main()