# PLAN DÉTAILLÉ - Transformation en Agent ReAct Décisionnel

## Ordre des tâches (1 commit par tâche, tests + mesure après chaque)

---

### TÂCHE 0 : GOLDEN SET ÉTENDU + BASELINE MESURÉE ✅ DÉJÀ FAIT
- `dataset_eval.jsonl` : 93 questions avec `passages_attendus` + `type_question` (simple/chiffree/multi-hop/sans_reponse/ambigue)
- `evaluate_extended.py` : Recall@5, MRR, nDCG@10, citation_accuracy, abstention_correct, latency p95, tokens/query
- **Action** : Lancer `evaluate_extended.py --skip-llm-eval` pour mesurer la baseline **architecture actuelle** (flags config.py par défaut) → sauvegarder `resultats_baseline_archi_actuelle.jsonl`

---

### TÂCHE 1 : SORTIES STRUCTURÉES ROBUSTES (déjà 90% fait)
**Objectif** : Taux sorties invalides < 1% sur golden set

**Existant** : `supervisor.py` a déjà `_parse_and_validate()` avec Pydantic + retry 2 + `format="json"` Ollama
**Manquant** : Appliquer le même pattern à TOUS les appels LLM structurés (reformulateur, futurs outils ReAct, claim checker)

**Actions** :
1. Créer `utils/structured_output.py` : fonction générique `call_llm_structured(prompt, schema, max_retries=2)` réutilisable
2. Refactorer `supervisor.py` et `nodes.py::reformulateur` pour l'utiliser
3. Ajouter métrique `invalid_output_rate` dans `evaluate_extended.py`
4. **Mesure** : Taux sorties invalides avant/après sur golden set

**Flag** : `ENABLE_STRUCTURED_OUTPUT` (déjà dans config.py)

---

### TÂCHE 2 : OUTIL CALCUL TABLEAUX (déjà implémenté ✅)
**Objectif** : Exactitude 100% sur questions chiffrées

**Existant** : `tools/table_calculator.py` complet + intégré dans agents contrats/assurances/maintenance via `_enrich_with_table_tool()`
**Flag** : `ENABLE_TABLE_TOOL` (déjà dans config.py)

**Actions** :
1. Vérifier que l'outil est bien appelé via LLM (pas auto-injecté) → **nécessite Tâche 3 ReAct**
2. Pour l'instant : mesure exactitude actuelle sur questions `type_question="chiffree"` (avec injection auto)
3. **Mesure** : Exactitude chiffrées avant/après activation flag

---

### TÂCHE 3 : BOUCLE REACT BORNÉE (CŒUR DU CHANGEMENT)
**Objectif** : Agent décide son chemin (outils, max 3 itérations, budgets)

#### 3.1 Nouveaux outils typés pour ReAct
| Outil | Fichier | Description |
|-------|---------|-------------|
| `search_hybrid` | Existant (`hybrid_retrieval.py`) | Retrieval BM25+Vector+RRF+Rerank |
| `text2cypher` | **NOUVEAU** `tools/text2cypher.py` | Wrapper `graph_query.py` avec schéma Pydantic |
| `calculate_table` | Existant (`tools/table_calculator.py`) | Calcul déterministe franchises/tarifs/seuils |
| `ask_clarification` | **NOUVEAU** `tools/clarification.py` | Demande précision utilisateur (retourne question clarifiée) |
| `final_answer` | **NOUVEAU** `tools/final_answer.py` | Termine la boucle, retourne réponse + sources |

#### 3.2 Schémas Pydantic (déjà dans `schemas.py`)
- `ToolCall(tool, args)`, `ToolResult(tool, success, result, error)`
- `ReActStep(step, tool, args, result, tokens_used)`
- `ReActState(steps[], total_tokens, start_time, finished, final_answer, abstention_reason)`

#### 3.3 Nœud ReAct unique (remplace 5 agents linéaires)
```python
def agent_react(state):
    # 1. Construire prompt ReAct avec outils disponibles + historique steps
    # 2. Appel LLM structured_output -> ToolCall
    # 3. Exécuter outil -> ToolResult
    # 4. Accumuler step dans ReActState
    # 5. Vérifier budgets (max 3 steps, tokens < 4000, temps < 30s)
    # 6. Si final_answer ou budget dépassé -> terminer
    # 7. Sinon -> loop
```

#### 3.4 Nouveau graphe LangGraph (`graph/builder.py`)
```
reformulateur → superviseur → [agent_react (boucle)] → END
                    ↘ hors_sujet
```

#### 3.5 Intégration flags
- `ENABLE_REACT=False` : comportement actuel exact (5 agents linéaires)
- `ENABLE_REACT=True` : boucle ReAct unique

#### 3.6 Mesures ablation
- Précision choix d'outil (outil attendu vs choisi)
- Nombre moyen d'étapes
- Taux atteinte limite (3 steps / tokens / temps)
- Comparaison métriques globales avec/sans REACT

---

### TÂCHE 4 : VÉRIFICATEUR DE CLAIMS
**Objectif** : % claims soutenus par preuves > 95%, hallucinations évitées

#### 4.1 Nouveau module `tools/claim_checker.py`
```python
def check_claims(response: str, contexts: List[str]) -> List[ClaimVerdict]:
    # 1. Découper réponse en affirmations atomiques (LLM ou règles)
    # 2. Pour chaque claim : prompt LLM-judge (Mistral, temp=0) + contexte
    # 3. Verdict Pydantic: supported(bool), evidence[], confidence
    # 4. Si TOUT supported -> OK
    # 5. Sinon : 1 régénération max avec feedback claims échoués
    # 6. Sinon -> abstention explicite "Je ne peux pas confirmer X"
```

#### 4.2 Intégration dans boucle ReAct
- Appelé automatiquement après `final_answer` si `ENABLE_CLAIM_CHECKER=True`
- Ajoute étape de vérification avant retour final

#### 4.3 Mesures
- % claims soutenus par preuves
- Exactitude citations
- Hallucinations évitées (comparaison vs sans claim checker)
- Coût latence (appels LLM supplémentaires)

**Flag** : `ENABLE_CLAIM_CHECKER` (déjà dans config.py)

---

### TÂCHE 5 : TRACES + SCRIPT ABLATION AUTOMATISÉ

#### 5.1 Traces (Langfuse self-hosted ou fallback JSONL)
**Préférence** : JSONL structuré (100% local, pas de dépendance lourde, NDA-safe)
```json
{"trace_id": "uuid", "step": "route", "decision": {"agents": ["contrats"]}, "latency_ms": 45}
{"trace_id": "uuid", "step": "tool_call", "tool": "search_hybrid", "args": {...}, "result_count": 5, "latency_ms": 120}
{"trace_id": "uuid", "step": "tool_call", "tool": "text2cypher", "args": {...}, "success": true, "latency_ms": 80}
{"trace_id": "uuid", "step": "claim_check", "claims": [...], "all_supported": true, "latency_ms": 200}
{"trace_id": "uuid", "step": "final", "abstention": false, "total_tokens": 1200, "total_time_sec": 2.3}
```

#### 5.2 Script `run_ablation.py`
```bash
# Exécute golden set avec chaque config :
# 1. Baseline (tout False)
# 2. + STRUCTURED_OUTPUT
# 3. + TABLE_TOOL
# 4. + REACT
# 5. + CLAIM_CHECKER
# 6. Tout True (full)
```
**Sortie** : Tableau Markdown `ablation_results.md`
| Métrique | Baseline | +Struct | +Table | +ReAct | +Claims | Full | Delta Total |
|----------|----------|---------|--------|--------|---------|------|-------------|
| Faithfulness | 0.82 | 0.83 | 0.83 | 0.85 | 0.87 | 0.88 | +0.06 |
| Recall@5 | 0.71 | 0.71 | 0.71 | 0.78 | 0.78 | 0.80 | +0.09 |
| Routing Acc | 0.88 | 0.89 | 0.89 | 0.91 | 0.91 | 0.92 | +0.04 |
| Chiffrées Acc | 0.65 | 0.65 | 0.92 | 0.92 | 0.92 | 0.95 | +0.30 |
| Latence p95 | 3.2s | 3.3s | 3.4s | 4.1s | 5.2s | 5.5s | +2.3s |
| Tokens/req | 800 | 820 | 850 | 1200 | 1500 | 1600 | +800 |

#### 5.3 Métriques de trajectoire (dans `evaluate_extended.py`)
- `tool_choice_accuracy` : bon outil choisi au bon moment
- `react_steps` : nombre étapes moyennes
- `token_cost_per_step` : budget respecté

---

### TÂCHES OPTIONNELLES (SI TEMPS)

| # | Composant | Description |
|---|-----------|-------------|
| 6 | **Planner** | Décomposition multi-hop en sous-questions (avant ReAct) |
| 7 | **CRAG Reflection** | Noter échec recherche, reformuler (2 tours max) |
| 8 | **Human Escalation** | `interrupt` LangGraph + checkpointing |
| 9 | **Calibration seuils** | Répondre / Reformuler / Abstenir sur jeu de test |
| 10 | **Model Routing** | Mistral pour routage/outils, modèle plus fort (ex: Llama 3.1 70B) pour réponse finale |

---

## RÈGLES DE DÉVELOPPEMENT (Rappel)

1. **Un ajout à la fois** → Tests → Mesure → Commit séparé
2. **Flags config.py** pour ablation (OBLIGATOIRE)
3. **Ne rien casser** : `evaluate_ragas.py`, `evaluate_baseline.py`, `dataset_eval.jsonl`, `main.py`, `api.py` doivent fonctionner identiquement
4. **Données = documents**, jamais instructions
5. **Pas de chiffres inventés** - uniquement mesures réelles
6. **Abstention** en cas de doute

---

## COMMANDES DE TEST APRÈS CHAQUE TÂCHE

```bash
# Tests régression (doivent tous passer)
python test_regression.py

# Évaluation rapide (sans LLM judge)
python evaluate_extended.py --skip-llm-eval --max-questions 10

# Évaluation complète (avec LLM judge) - pour mesure finale
python evaluate_extended.py --max-questions 20

# Baseline comparaison
python evaluate_baseline.py --skip-llm-eval --max-questions 10
```

---

## VALIDATION REQUISE

**Ce plan te convient-il ?** 
- Si OUI : je commence par Tâche 1 (Structured Output utilitaire + métrique)
- Si MODIFS : dis-moi quoi changer

Une fois validé, j'implémenterai **Tâche 1** en premier.