# Explainable-Neurosymbolic-IDS

Prototype académique de détection d’intrusions explicable combinant apprentissage automatique, SHAP, transformation sémantique, raisonnement symbolique avec Scallop, MITRE ATT&CK, reconstruction de chaîne d’attaque et génération de preuves explicables.

> Projet de recherche de maîtrise en informatique à l’Université Laval. Ce dépôt est expérimental et n’est pas destiné à un déploiement direct en production.

## Objectif

Le projet transforme une prédiction issue d’un modèle d’apprentissage automatique en une explication exploitable par un analyste SOC. Le pipeline produit notamment :

- une prédiction et une probabilité ;
- une explication locale SHAP ;
- des concepts et actions sémantiques ;
- des motifs comportementaux activés par Scallop ;
- des candidats MITRE ATT&CK ;
- une chaîne d’attaque ;
- une preuve explicable ;
- un niveau de risque, une priorité SOC et une recommandation.

## Classes prises en charge

Le prototype utilise les classes NSL-KDD suivantes :

- `Normal`
- `DoS`
- `Probe`
- `R2L`
- `U2R`

## Pipeline

```text
Entrée XAI / CoMAT
        |
        v
Prédiction du modèle
        |
        v
Explication locale SHAP
        |
        v
Feature -> Concept
        |
        v
Concept -> Action
        |
        v
Raisonnement Scallop
        |
        v
Motifs comportementaux
        |
        v
MITRE ATT&CK
        |
        v
Chaîne d’attaque
        |
        v
Preuve explicable et recommandation SOC
```

Le pipeline principal suit :

```text
XAI/CoMAT -> Concepts -> Actions -> Scallop -> MITRE ATT&CK
          -> Chaîne d’attaque -> Preuve E=<O,R,J>
```

## Preuve explicable

La preuve suit la structure :

```text
E = <O, R, J>
```

où `O` représente les observations traçables, `R` la relation ou règle explicative, et `J` la justification interprétable et vérifiable.

## Architecture

```text
.
├── main.py
├── requirements.txt
├── pyproject.toml
├── config/
│   ├── comat/
│   ├── mitre/
│   ├── reasoning/
│   └── semantic/
├── docs/
│   └── SCALLOP_INSTALL.md
├── knowledge_base/
│   ├── actions/
│   ├── concepts/
│   └── patterns/
├── models/
│   ├── feature_names.joblib
│   ├── preprocessor.joblib
│   └── random_forest.joblib
├── outputs/
│   ├── comat/
│   ├── experiments/
│   ├── mitre/
│   ├── proofs/
│   ├── reasoning/
│   ├── semantic/
│   └── shap/
├── scallop/
│   └── behavioral_reasoning.scl
├── scripts/
├── src/
│   ├── detection/
│   ├── explanation/
│   ├── integration/
│   ├── knowledge/
│   ├── mitre/
│   ├── reasoning/
│   ├── semantic/
│   ├── utils/
│   └── xai/
└── tests/
```

## Composants

### Détection

`src/detection/` contient le prétraitement et l’utilisation du modèle Random Forest. Les artefacts sont stockés dans `models/`.

### Explicabilité SHAP

`src/xai/` calcule et exporte les explications locales utilisées par la suite du pipeline.

### Intégration CoMAT

`src/integration/comat_adapter.py` adapte les sorties XAI au format d’entrée du pipeline. Des exemples sont disponibles dans `outputs/comat/`.

### Transformation sémantique

`src/semantic/mapper.py` réalise :

```text
Feature -> Concept
Concept -> Action
```

Les configurations sont dans `config/semantic/` et les connaissances dans `knowledge_base/`.

### Raisonnement symbolique

Le backend principal est `src/reasoning/scallop_backend.py`. Le programme logique est `scallop/behavioral_reasoning.scl`.

### MITRE ATT&CK

`src/mitre/mapper.py` transforme les motifs activés en candidats MITRE ATT&CK à partir de `config/mitre/pattern_to_mitre.json`.

### Chaîne d’attaque

`src/reasoning/attack_chain_builder.py` ordonne les candidats selon les tactiques MITRE ATT&CK.

### Preuve explicable

`src/explanation/proof_builder.py` construit la preuve finale à partir de la prédiction, des actions, du raisonnement, des candidats MITRE et de la chaîne d’attaque.

## Installation

```bash
git clone <URL_DU_DEPOT>
cd Explainable-Neurosymbolic-IDS
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Installation de Scallopy

Scallopy est installé séparément à partir d’un wheel compatible. Voir :

```text
docs/SCALLOP_INSTALL.md
```

Exemple :

```bash
python -m pip install /chemin/vers/scallopy-0.2.5-cp312-cp312-macosx_11_0_arm64.whl
```

## Exécution

Exemple validé :

```bash
python main.py --input outputs/comat/input_probe_01.json
```

Le programme déduit l’identifiant du cas depuis le nom du fichier, par exemple :

```text
input_probe_01.json -> probe_01
```

## Sorties

Pour un identifiant `<case_id>`, le pipeline produit :

```text
outputs/semantic/concepts_<case_id>.json
outputs/semantic/actions_<case_id>.json
outputs/reasoning/reasoning_<case_id>.json
outputs/mitre/mitre_<case_id>.json
outputs/reasoning/attack_chain_<case_id>.json
outputs/proofs/proof_<case_id>.json
```

Les sorties SHAP sont dans `outputs/shap/`.

## Cas disponibles

```text
outputs/comat/input_normal_01.json
outputs/comat/input_dos_01.json
outputs/comat/input_probe_01.json
outputs/comat/input_r2l_01.json
outputs/comat/input_u2r_01.json
outputs/comat/input_case_0.json
```

## Tests

```bash
pytest
```

État validé :

```text
7 passed
```

Les tests couvrent notamment le mapper MITRE, la chaîne d’attaque, le pipeline réel et le backend Scallop.

## Scripts principaux

```text
scripts/train_random_forest.py
scripts/export_shap_case.py
scripts/run_explainable_pipeline.py
scripts/run_batch_experiments.py
scripts/compare_random_forests.py
```

Le dossier `scripts/experiments/` contient les scripts de statistiques, d’export, de discussion, de figures et de génération LaTeX.

## Reproductibilité

Le projet a été validé dans un environnement Python 3.12 propre :

```bash
python3.12 -m venv .venv-test
source .venv-test/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip install /chemin/vers/le/wheel/scallopy.whl
pytest
python main.py --input outputs/comat/input_probe_01.json
```

Résultats observés :

- dépendances standards importées ;
- Scallopy importé ;
- pipeline complet exécuté ;
- sorties générées ;
- 7 tests réussis.

## Limites

- Prototype expérimental.
- Les résultats MITRE ATT&CK sont des candidats à valider avec le contexte SOC.
- SHAP explique le comportement du modèle, mais n’établit pas seul une causalité.
- La qualité dépend du modèle, des données, des règles et de la base de connaissances.
- La criticité des actifs et le contexte métier restent nécessaires.
- Scallopy nécessite une installation séparée dépendante de la plateforme.

## Auteur

**Brunel Gafoube Fotso**  
Maîtrise recherche en informatique  
Université Laval

## Licence

Aucune licence open source active n’est actuellement définie. Tous droits réservés.
