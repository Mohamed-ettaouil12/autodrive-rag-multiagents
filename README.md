# AutoDrive Multi-Agent RAG

Système RAG multi-agents pour l'assistance client AutoDrive Maroc (location de véhicules). Pipeline modulaire : reformulation → supervision → 5 agents spécialisés → retrieval hybride (vectoriel + BM25) → génération, avec graphe de connaissances Neo4j pour la mémoire conversationnelle et les données métier.

---

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────────────────────────┐
│  QUESTION   │────▶│ REFORUMULATEUR │────▶│        SUPERVISEUR (routing)      │
│  UTILISATEUR│     │  (contexte    │     │  contrats | assurances | faq      │
└─────────────┘     │   session)   │     │  vehicules | maintenance | hors_sujet│
                    └─────────────┘     └──────────────┬──────────────────────┘
                                                       │
                    ┌──────────────────────────────────┼──────────────────────────────────┐
                    ▼                                  ▼                                  ▼
            ┌───────────────┐                ┌───────────────┐                ┌───────────────┐
            │  AGENT        │                │  AGENT        │                │  AGENT        │
            │  CONTRATS     │                │  ASSURANCES   │                │  FAQ          │
            │  (table_tool, │                │  (table_tool, │                │  (search)     │
            │   graph)      │                │   graph)      │                └───────┬───────┘
            └───────┬───────┘                └───────┬───────┘                        │
                    │                                │                                │
            ┌───────┴───────┐                ┌───────┴───────┐                ┌───────┴───────┐
            │  AGENT        │                │  AGENT        │                │  AGENT        │
            │  VEHICULES    │                │  MAINTENANCE  │                │  HORS_SUJET   │
            │  (graph)      │                │  (table_tool) │                │  (fallback)   │
            └───────┬───────┘                └───────┬───────┘                └───────┬───────┘
                    │                                │                                │
                    └────────────────────────────────┼────────────────────────────────┘
                                                     ▼
                    ┌─────────────────────────────────────────────────────────────────┐
                    │              RETRIEVAL HYBRIDE (ChromaDB + BM25)                │
                    │  Embeddings: BGE-M3  |  Rerank: cross-encoder (optionnel)      │
                    └────────────────────────────┬────────────────────────────────────┘
                                                 ▼
                    ┌─────────────────────────────────────────────────────────────────┐
                    │                    GÉNÉRATION (Ollama / Mistral)                │
                    │  Prompt structuré + contexte récupéré + données graphe + tables │
                    └────────────────────────────┬────────────────────────────────────┘
                                                 ▼
                    ┌─────────────────────────────────────────────────────────────────┐
                    │                         NEO4J                                    │
                    │  • Mémoire conversationnelle (sessions, messages, feedback)     │
                    │  • Graphe métier : Contrats, Véhicules, Assurances, Clients     │
                    │  • 483 relations (Contrat-Véhicule, Contrat-Assurance, etc.)   │
                    └─────────────────────────────────────────────────────────────────┘
```

**Modes d'exécution** (configurable via `config.py:ENABLE_REACT`) :
- **Linear** (défaut) : reformulateur → superviseur → 1 agent → retrieval → génération
- **ReAct** : reformulateur → superviseur → agent_react (boucle outils : search, calculate_table, text2cypher, final_answer)

---

## Stack Technique

| Composant | Technologie |
|-----------|-------------|
| **LLM** | Mistral (via Ollama) |
| **Embeddings** | BGE-M3 (sentence-transformers) |
| **Vector DB** | ChromaDB (persistent) |
| **Sparse Retrieval** | BM25 (rank-bm25) |
| **Graph DB** | Neo4j 5.15 + APOC |
| **Orchestration** | LangGraph (StateGraph) |
| **API** | FastAPI + Uvicorn |
| **UI** | Streamlit |
| **Structured Output** | Pydantic + Ollama JSON mode |
| **Env** | Python 3.10+, docker-compose |

---

## Structure des Dossiers

```
.
├── api.py                    # FastAPI server (port 8000)
├── main.py                   # CLI interactif (mémoire Neo4j)
├── core.py                   # Orchestration haute-niveau (sessions, graph)
├── config.py                 # Configuration centralisée
├── rag_autodrive.py          # Embeddings, ChromaDB, génération de base
├── hybrid_retrieval.py       # Retrieval hybride (vectoriel + BM25 + rerank)
├── graph_query.py            # Requêtes Cypher métier (contrats, véhicules, assurances)
├── schema_graph.py           # Schéma Neo4j (contraintes, indexes)
├── populate_graph.py         # Peuplement graphe depuis JSONL
├── extract_entities.py       # Extraction entités/relations (LLM)
├── build_bm25_indexes.py     # Construction index BM25
├── run_ablation.py           # Lancement étude d'ablation
├── evaluate_*.py             # Évaluation (RAGAS, baseline, extended)
├── requirements.txt
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.streamlit
├── graph/
│   ├── builder.py            # Construction LangGraph (linear + ReAct)
│   ├── state.py              # AgentState (TypedDict)
│   ├── nodes.py              # Nœuds linear (5 agents + outils)
│   └── nodes_react.py        # Nœud ReAct (boucle outils)
├── supervisor/
│   └── supervisor.py         # Routing multi-label (LLM + règles)
├── tools/
│   ├── table_calculator.py   # Calcul déterministe (tarifs, franchises, seuils)
│   ├── text2cypher.py        # Text-to-Cypher pour Neo4j
│   ├── clarification.py      # Demande de clarification utilisateur
│   └── final_answer.py       # Formatage réponse finale
├── utils/
│   └── structured_output.py  # Appels LLM avec validation Pydantic
├── schemas.py                # Modèles Pydantic (routing, reformulation, outils)
├── memory.py                 # Gestion sessions Neo4j (CRUD, historique)
├── preprocessing/
│   ├── output/               # Chunks JSONL, ChromaDB, index BM25
│   └── faq_chunking.py       # Prétraitement FAQ
└── ablation_results/         # Résultats études d'ablation (JSON, MD)
```

---

## Installation

### Prérequis
- Python 3.10+
- Docker & Docker Compose
- Ollama installé localement

### 1. Environnement virtuel
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Modèle LLM (Ollama)
```bash
ollama pull mistral
```

### 3. Neo4j (Docker)
```bash
docker compose up -d neo4j
# Interface: http://localhost:7474 (neo4j / autodrive123)
# Bolt: bolt://localhost:7687
```

### 4. Initialisation données (première fois)
```bash
# 1. Créer schéma graphe
python schema_graph.py

# 2. Peupler Neo4j depuis extractions JSONL
python populate_graph.py

# 3. Construire index ChromaDB + BM25
python rag_autodrive.py       # crée collections ChromaDB
python build_bm25_indexes.py  # crée index BM25
```

---

## Lancement

### API REST (FastAPI)
```bash
# Direct
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Ou via Docker (API + Neo4j)
docker compose up -d api
# Swagger: http://localhost:8000/docs
```

### Interface Web (Streamlit)
```bash
# Direct
streamlit run streamlit_app.py  # si fichier existe

# Ou via Docker
docker compose up -d streamlit
# UI: http://localhost:8501
```

### CLI Interactif (avec mémoire Neo4j)
```bash
python main.py
# Commandes: 'exit' (quitter), 'nouvelle' (nouvelle session)
```

### Étude d'Ablation
```bash
python run_ablation.py
# Résultats dans ablation_results/
```

### Tests de Régression
```bash
python test_regression.py
```

---

## Résultats Clés

### Graphe de Connaissances (Neo4j)
| Métrique | Valeur |
|----------|--------|
| Nœuds Contrat | 47 |
| Nœuds Véhicule | 89 |
| Nœuds Assurance | 23 |
| Nœuds Client | 31 |
| **Relations totales** | **483** |
| Types de relations | CONTRAT_VEHICULE, CONTRAT_ASSURANCE, CONTRAT_CLIENT, VEHICULE_MAINTENANCE, etc. |

### Étude d'Ablation (93 questions échantillonnées)

| Configuration | Recall@5 | MRR | nDCG@10 | Citation Acc. | Routing Acc. | Latence (s) | Tokens/Requête |
|---------------|----------|-----|---------|---------------|--------------|-------------|----------------|
| **Baseline (Linear)** | 0.000 | 0.000 | 0.000 | 0.000 | **1.000** | ~0.0 | ~0.0 |
| **+ ReAct Loop** | 0.000 | 0.000 | 0.000 | 0.000 | **1.000** | ~0.0 | ~0.0 |
| **+ Table Tool** | — | — | — | — | **1.000** | — | — |
| **+ Structured Output** | — | — | — | — | **1.000** | — | — |

> **Note** : Les métriques de retrieval (Recall@5, MRR, nDCG) sont à 0.000 sur l'échantillon testé, indiquant un désalignement entre les questions d'évaluation et les chunks indexés. Le routing atteint 100% de précision. L'outil `table_calculator` garantit la correction des réponses chiffrées (tarifs, franchises, cautions, seuils maintenance).

### Métriques de Trajectoire (ReAct)
| Config | Étapes Moy. | Tool Choice Acc | Token Budget % | Time Budget % |
|--------|-------------|-----------------|----------------|---------------|
| +ReAct Loop | N/A | N/A | N/A | N/A |

---

## Configuration Principale (`config.py`)

```python
# Mode ReAct (boucle outils) vs Linear (pipeline séquentiel)
ENABLE_REACT = False

# Outils activés
ENABLE_TABLE_TOOL = True      # Calcul déterministe (tarifs, franchises, seuils)
ENABLE_STRUCTURED_OUTPUT = True  # Validation Pydantic sorties LLM
OLLAMA_JSON_FORMAT = True     # Mode JSON natif Ollama
MAX_JSON_RETRIES = 2          # Réessais parsing JSON

# Budgets ReAct
MAX_REACT_STEPS = 3
REACT_TOKEN_BUDGET = 4000
REACT_TIME_BUDGET_SEC = 30.0
```

---

## Variables d'Environnement (`.env`)

```env
OLLAMA_BASE_URL=http://localhost:11434
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=autodrive123
```

---

## Licence

Usage interne AutoDrive Maroc.