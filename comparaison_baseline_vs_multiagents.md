# Comparaison Baseline vs Multi-Agents

*Date: 2026-10-10 21:10*

## Métriques Globales

| Métrique | Baseline | Multi-Agents | Delta | Cible |
|----------|----------|--------------|-------|-------|
| faithfulness | 0.9231 | 0.9303 | +0.0072 | 0.87 |
| answer_relevancy | 1.0000 | 0.8829 | -0.1171 | 0.82 |
| context_precision | 0.7885 | 0.6579 | -0.1306 | 0.78 |
| routing_accuracy | 0.0% | 75.0% | +75.0% | - |
| avg_time_sec | 6.9 | 13.9 | +7.0 | - |

## Par Catégorie

| Catégorie | Système | Faithfulness | Answer Relevancy | Context Precision | Routing | N |
|-----------|---------|--------------|------------------|-------------------|---------|---|
| assurances | Baseline | 0.000 | 0.000 | 0.000 | 0/14 | 14 |
| assurances | Multi-Agents | 1.000 | 0.892 | 0.417 | 12/12 | 12 |
| contrats | Baseline | 0.000 | 0.000 | 0.000 | 0/10 | 10 |
| contrats | Multi-Agents | 0.800 | 0.600 | 0.600 | 15/15 | 15 |
| faq | Baseline | 0.000 | 0.000 | 0.000 | 0/12 | 12 |
| faq | Multi-Agents | 0.917 | 0.987 | 0.750 | 8/12 | 12 |
| maintenance | Baseline | 1.000 | 1.000 | 0.750 | 0/10 | 10 |
| maintenance | Multi-Agents | 0.000 | 0.000 | 0.000 | 4/10 | 10 |
| vehicules | Baseline | 0.900 | 1.000 | 0.800 | 0/11 | 11 |
| vehicules | Multi-Agents | 0.928 | 0.889 | 0.889 | 6/11 | 11 |

## Par Question

| ID | Question | Catégorie | Baseline Faith. | Multi Faith. | Delta | Baseline Rel. | Multi Rel. | Delta |
|----|----------|-----------|-----------------|--------------|-------|---------------|------------|-------|
| Q001 | ... |  | 0.000 | 1.000 | +1.000 | 0.000 | 0.500 | +0.500 |
| Q002 | ... |  | 0.000 | 1.000 | +1.000 | 0.000 | 1.000 | +1.000 |
| Q003 | ... |  | 0.000 | 1.000 | +1.000 | 0.000 | 1.000 | +1.000 |
| Q004 | ... |  | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| Q005 | ... |  | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| Q006 | Comment est calculée la caution pour un ... | contrats | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| Q007 | Quel est le kilométrage inclus pour le c... | contrats | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| Q008 | Quelle est la règle carburant pour le co... | contrats | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| Q009 | Quelles sont les pièces requises pour le... | contrats | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| Q010 | Quel est le tarif journalier du contrat ... | contrats | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| Q011 | Quelle est la règle de retard pour le co... | contrats | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| Q012 | Quel est l'âge minimum pour le contrat R... | contrats | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| Q013 | Quelle est la caution pour le contrat RA... | contrats | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| Q014 | Quel est le kilométrage inclus pour le c... | contrats | 0.000 | 0.000 | +0.000 | 0.000 | 0.500 | +0.500 |
| Q015 | Quelles sont les pièces requises pour le... | contrats | 0.000 | 1.000 | +1.000 | 0.000 | 0.000 | +0.000 |
| Q016 | Quelles sont les couvertures de l'assura... | assurances | 0.000 | 1.000 | +1.000 | 0.000 | 1.000 | +1.000 |
| Q017 | Quelles sont les exclusions de l'assuran... | assurances | 0.000 | 1.000 | +1.000 | 0.000 | 1.000 | +1.000 |
| Q018 | Quelle est la franchise de l'assurance B... | assurances | 0.000 | 1.000 | +1.000 | 0.000 | 0.850 | +0.850 |
| Q019 | L'assurance Basic couvre-t-elle le bris ... | assurances | 0.000 | 1.000 | +1.000 | 0.000 | 1.000 | +1.000 |
| Q020 | L'assurance Basic couvre-t-elle le vol d... | assurances | 0.000 | 1.000 | +1.000 | 0.000 | 1.000 | +1.000 |
| Q021 | Quelles sont les couvertures de l'assura... | assurances | 0.000 | 1.000 | +1.000 | 0.000 | 1.000 | +1.000 |
| Q022 | Quelles sont les exclusions de l'assuran... | assurances | 0.000 | 1.000 | +1.000 | 0.000 | 1.000 | +1.000 |
| Q023 | Quelle est la franchise de l'assurance B... | assurances | 0.000 | 1.000 | +1.000 | 0.000 | 0.500 | +0.500 |
| Q024 | L'assurance Basic couvre-t-elle le bris ... | assurances | 0.000 | 1.000 | +1.000 | 0.000 | 1.000 | +1.000 |
| Q025 | L'assurance Basic couvre-t-elle le vol d... | assurances | 0.000 | 1.000 | +1.000 | 0.000 | 0.350 | +0.350 |
| Q026 | Quelles sont les couvertures de l'assura... | assurances | 0.000 | 1.000 | +1.000 | 0.000 | 1.000 | +1.000 |
| Q027 | Quelles sont les exclusions de l'assuran... | assurances | 0.000 | 1.000 | +1.000 | 0.000 | 1.000 | +1.000 |
| Q028 | Quelles agences AutoDrive Maroc sont dis... | faq | 0.000 | 1.000 | +1.000 | 0.000 | 1.000 | +1.000 |
| Q029 | Quels sont les horaires de l'agence de C... | faq | 0.000 | 1.000 | +1.000 | 0.000 | 1.000 | +1.000 |
| Q030 | Quels sont les horaires de l'agence de R... | faq | 0.000 | 1.000 | +1.000 | 0.000 | 1.000 | +1.000 |
| Q031 | Quels sont les horaires de l'agence de M... | faq | 0.000 | 1.000 | +1.000 | 0.000 | 1.000 | +1.000 |
| Q032 | Quelles catégories de véhicules peut-on ... | faq | 0.000 | 1.000 | +1.000 | 0.000 | 1.000 | +1.000 |
| Q033 | Combien de places offre un monospace de ... | faq | 0.000 | 1.000 | +1.000 | 0.000 | 1.000 | +1.000 |
| Q034 | Un conducteur de 22 ans avec trois ans d... | faq | 0.000 | 1.000 | +1.000 | 0.000 | 1.000 | +1.000 |
| Q035 | Un conducteur de 23 ans avec seulement d... | faq | 0.000 | 0.000 | +0.000 | 0.000 | 1.000 | +1.000 |
| Q036 | Une personne de 21 ans avec deux ans de ... | faq | 0.000 | 1.000 | +1.000 | 0.000 | 1.000 | +1.000 |
| Q037 | Une entreprise peut-elle réserver un véh... | faq | 0.000 | 1.000 | +1.000 | 0.000 | 1.000 | +1.000 |
| Q038 | Un touriste étranger peut-il réserver av... | faq | 0.000 | 1.000 | +1.000 | 0.000 | 0.850 | +0.850 |
| Q039 | Je veux une voiture automatique pour cin... | faq | 0.000 | 1.000 | +1.000 | 0.000 | 1.000 | +1.000 |
| Q040 | Quelle est la catégorie du Dacia Sandero... | vehicules | 0.000 | 1.000 | +1.000 | 0.000 | 1.000 | +1.000 |
| Q041 | Quelle transmission pour le Dacia Sander... | vehicules | 1.000 | 0.350 | -0.650 | 1.000 | 0.000 | -1.000 |
| Q042 | Combien de places pour le Dacia Sandero ... | vehicules | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| Q043 | Quel est le tarif journalier de la catég... | vehicules | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| Q044 | À quelle fréquence la vidange pour le Da... | vehicules | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| Q045 | Quelle est la catégorie du Renault Clio ... | vehicules | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| Q046 | Quelle transmission pour le Renault Clio... | vehicules | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| Q047 | Combien de places pour le Renault Clio ?... | vehicules | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| Q048 | Quel est le tarif journalier de la catég... | vehicules | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| Q049 | À quelle fréquence la vidange pour le Re... | vehicules | 0.000 | 0.000 | +0.000 | 1.000 | 0.000 | -1.000 |
| Q050 | Quelle est la catégorie du Hyundai i10 (... | vehicules | 1.000 | 0.000 | -1.000 | 1.000 | 0.000 | -1.000 |
| Q051 | Quels checkpoints doivent être repris av... | maintenance | 1.000 | 0.000 | -1.000 | 1.000 | 0.000 | -1.000 |
| Q052 | À quel kilométrage la vidange est-elle p... | maintenance | 1.000 | 0.000 | -1.000 | 1.000 | 0.000 | -1.000 |
| Q053 | Quels sont les seuils pneus et freinage ... | maintenance | 1.000 | 0.000 | -1.000 | 1.000 | 0.000 | -1.000 |
| Q054 | Le client signale un voyant rouge juste ... | maintenance | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| Q055 | Quand l’urgence est-elle critique pendan... | maintenance | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| Q056 | Quelles sont les étapes de inspection av... | maintenance | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| Q057 | Quels symptômes pour inspection avant lo... | maintenance | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| Q058 | Quels checkpoints doivent être repris au... | maintenance | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| Q059 | Quel est le seuil de batterie pour la ca... | maintenance | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| Q060 | Quel est le seuil de climatisation pour ... | maintenance | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |