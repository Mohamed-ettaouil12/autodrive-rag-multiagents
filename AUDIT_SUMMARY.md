# AUDIT DU CODE EXISTANT - AutoDrive Multi-Agent RAG

## Ce qui existe déjà (Résumé en 10 lignes)

1. **Architecture linéaire LangGraph** : `reformulateur` → `superviseur` (routage JSON vers 5 agents) → agents RAG hybrides (BM25 + ChromaDB + RRF + cross-encoder) → génération Mistral 7B
2. **Superviseur avec sorties structurées** : Pydantic `RoutingDecision`, décodage contraint Ollama `format="json"`, retry borné (2 max), validation agents
3. **Retrieval hybride complet** : `hybrid_retrieval.py` avec BM25 (index pickle), vectoriel ChromaDB, fusion RRF (k=60), reranking cross-encoder ms-marco-MiniLM-L6-v2
4. **Graphe Neo4j branché par regex** : `graph_query.py` avec 4 requêtes Cypher (contrat, assurance contrat, historique véhicule, recherche entité), déclenché dans `nodes.py` si pattern contrat/immat détecté
5. **Outil table_calculator déterministe** : `tools/table_calculator.py` extrait valeurs numériques (tarifs, franchises, cautions, seuils maintenance) depuis chunks entités + regex sur FAQ, appelé automatiquement dans agents contrats/assurances/maintenance
6. **Évaluation complète** : 
   - `dataset_eval.jsonl` (93 questions avec `passages_attendus`, `type_question`: simple/chiffree/multi-hop/sans_reponse/ambigue)
   - `evaluate_ragas.py` (faithfulness/relevancy/precision + routing accuracy via Ollama-judge)
   - `evaluate_extended.py` (+ Recall@5, MRR, nDCG@10, citation_accuracy, abstention_correct, latency p95, tokens/query)
   - `evaluate_baseline.py` (comparaison baseline vectoriel simple vs multi-agents)
7. **Flags d'ablation dans config.py** : `ENABLE_STRUCTURED_OUTPUT`, `ENABLE_TABLE_TOOL`, `ENABLE_REACT`, `ENABLE_CLAIM_CHECKER` (déjà définis mais `ENABLE_REACT=False`, `ENABLE_CLAIM_CHECKER=False`)
8. **API FastAPI + Streamlit** : `/chat`, `/health`, `/sessions/{id}/historique`, `/feedback` avec mémoire Neo4j sessions/messages/feedback
9. **Tests de régression** : `test_regression.py` valide imports, supervisor, graph, hybrid retrieval, 3 scripts d'éval (avec `--skip-llm-eval`)
10. **Dégradation gracieuse** : Neo4j down → RAG seul, cross-encoder down → pas de reranking, BM25 index manquant → reconstruction à la volée

## Ce qui MANQUE pour l'objectif "Agent ReAct décide son chemin"

| Composant | État | Travail requis |
|-----------|------|----------------|
| **Boucle ReAct bornée** | ❌ Absent | Remplacer le nœud agent linéaire par boucle LangGraph avec outils typés, max 3 étapes, budget tokens/temps |
| **Outils typés pour ReAct** | Partiel | `search_hybrid` existe, `text2cypher` (wrapper graph_query.py), `calculate_table` existe, `ask_clarification`, `final_answer` à créer |
| **Vérificateur de claims** | ❌ Absent | Découpage réponse en affirmations, vérification LLM-judge (temp=0) contre preuves, 1 régénération max puis abstention |
| **Traces Langfuse/JSONL** | ❌ Absent | Une trace par décision (route, outil, score, abstention) |
| **Script d'ablation automatisé** | ❌ Absent | `run_ablation.py` : exécute golden set avec chaque flag on/off, produit tableau markdown comparatif |
| **Routage Neo4j piloté par agent** | Heuristique regex | Neo4j appelé SEULEMENT si l'agent choisit l'outil `text2cypher` (plus de regex forcée) |

## Fichiers clés à modifier (dans l'ordre)

1. `config.py` - flags déjà là, ajouter budgets ReAct/ClaimChecker
2. `schemas.py` - schémas Pydantic déjà définis (ToolCall, ToolResult, ReActStep, ReActState, ClaimVerdict)
3. `graph/state.py` - étendre `AgentState` pour état ReAct (steps, tokens, temps, finished, abstention)
4. `graph/nodes.py` - **cœur du changement** : remplacer agents linéaires par nœud ReAct unique + outils
5. `graph/builder.py` - nouveau graphe avec boucle conditionnelle ReAct
6. `tools/text2cypher.py` - NOUVEAU : wrapper typé `graph_query.py` pour ReAct
7. `tools/claim_checker.py` - NOUVEAU : vérificateur d'affirmations
8. `tools/clarification.py` - NOUVEAU : outil demande de clarification
9. `run_ablation.py` - NOUVEAU : script d'ablation complet
10. `evaluate_extended.py` - ajouter métriques trajectoire (bon outil, nb étapes, tokens)
11. Tests existants - doivent continuer à passer

## Validation requise avant de coder

Le plan détaillé ci-dessous est prêt. **Attends ton accord** pour commencer l'implémentation étape par étape (1 commit par ajout, tests après chaque étape).