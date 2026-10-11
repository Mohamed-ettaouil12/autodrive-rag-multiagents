#!/usr/bin/env python3
"""
Script pour construire les index BM25 pour toutes les catégories.
À exécuter une fois, ou après mise à jour des chunks.
"""

import logging
from hybrid_retrieval import build_all_bm25_indexes

LOG_FORMAT = "%(asctime)s [BUILD_BM25] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

if __name__ == "__main__":
    build_all_bm25_indexes()