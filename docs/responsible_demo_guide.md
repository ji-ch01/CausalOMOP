# Guide de démonstration responsable

Ce guide présente CausalOMOP comme un prototype de conception, de gouvernance
et de faisabilité. Il ne faut jamais présenter ses sorties actuelles comme une
estimation d'effet causal ou une recommandation clinique.

## Message d'ouverture

> CausalOMOP transforme une question clinique en protocole de target trial, DAG
> causal documenté, revue d'identification, mappings OMOP et audits de
> faisabilité traçables. Les sorties LLM restent des propositions, les décisions
> importantes sont humaines, et aucun estimateur causal n'est encore implémenté.

## 1. Vérifier l'environnement

Depuis la racine du dépôt :

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest tests -q
python -m causal.cli config --json
```

Avec BigQuery et les identifiants applicatifs configurés :

```powershell
python -m causal.cli preflight --count-persons
```

Ne pas annoncer un nombre de tests, de tables ou de personnes mémorisé dans ce
guide : montrer la sortie produite dans l'environnement de démonstration.

## 2. Créer un run

À partir du protocole versionné :

```powershell
$state = python -m causal.cli workflow start studies\albumin_sepsis_28d\protocol.json |
  ConvertFrom-Json
$runId = $state.run_id
python -m causal.cli workflow hypothesis $runId
```

Ou à partir d'une question naturelle :

```powershell
python -m causal.cli workflow start-question "Among adult ICU patients with sepsis, what is the effect of albumin plus crystalloids versus crystalloids alone on 28-day mortality?"
```

Expliquer le filtre initial : une question descriptive est renvoyée vers
FastOMOP et une question ambiguë demande une clarification. Le chemin causal
peut ensuite interroger PubMed et OpenAlex, utiliser OptimusKG s'il est
disponible, puis exécuter une critique bornée du DAG.

## 3. Montrer les artefacts auditables

Sous `artifacts/runs/<run_id>/`, montrer :

- `protocol.json` : target trial et mappings encore soumis à revue ;
- `hypothesis.json` : DAG, problèmes non résolus et identification ;
- `llm_outputs.json` : réponses brutes, tours de critique et révision ;
- `state.json` : phase, gates, reviewer et historique.

Souligner que `executable: false` est un garde-fou normal tant que les revues
requises ne sont pas terminées.

Exporter le DAG pour une inspection indépendante :

```powershell
python -m causal.cli workflow dag-export <run_id> --highlight-adjustment-set
```

Le texte peut être collé dans [dagitty.net](https://dagitty.net). Une
identification mathématique par backdoor n'établit pas à elle seule la
complétude clinique du DAG.

## 4. Faire une vraie revue humaine

Utiliser un identifiant de reviewer réel et un commentaire traçable :

```powershell
python -m causal.cli workflow gate <run_id> protocol approved `
  --reviewer "reviewer-id" `
  --comment "Target trial et DAG acceptés pour exploration technique"
python -m causal.cli workflow advance <run_id>
python -m causal.cli workflow plan <run_id> --with-mcp
```

Les candidats de vocabulaire renvoyés par OMCP ne sont jamais approuvés
automatiquement. Les mappings se font via `workflow concept-set-map`, les
commandes `workflow covariate-*`, ou l'interface Streamlit :

```powershell
streamlit run causal/ui/streamlit_app.py
```

Ne pas approuver un concept ou un gate uniquement pour faire réussir la démo.

## 5. Montrer le registre et les audits

Après les revues nécessaires :

```powershell
python -m causal.cli workflow registry-build <run_id>
python -m causal.cli workflow gate <run_id> clinical_mapping provisional `
  --reviewer "reviewer-id" --comment "Audits agrégés uniquement"
python -m causal.cli workflow advance <run_id>
python -m causal.cli workflow feasibility <run_id> --transport mcp
python -m causal.cli workflow execution-readiness <run_id>
python -m causal.cli workflow cohort-plan <run_id>
python -m causal.cli workflow cohort-profile <run_id>
python -m causal.cli workflow followup-audit <run_id>
python -m causal.cli workflow mapping-audit <run_id> <variable_id>
python -m causal.cli workflow demo-report <run_id>
```

Toutes les commandes ne réussiront que si l'état et les gates du run le
permettent. C'est un comportement attendu à montrer, pas un obstacle à
contourner.

## Résultats présentables

- un protocole et un DAG candidats, versionnés et auditables ;
- la provenance des preuves et les limites de la recherche bibliographique ;
- les décisions humaines et les blocages de gouvernance ;
- des mappings OMOP proposés ou approuvés, avec leur audit de couverture ;
- des comptages de faisabilité agrégés et explicitement datés ;
- un plan SQL de cohorte analytique qui n'implique pas son autorisation
  d'exécution ou de diffusion.

## Résultats non présentables

- un effectif provisoire comme cohorte finale ;
- le DAG généré comme vérité clinique ;
- une différence de mortalité comme effet du traitement ;
- un ATE, ATT ou CATE ;
- une conclusion clinique sur l'albumine ;
- une reproduction réussie de l'article de référence.

Le [concept audit](../studies/albumin_sepsis_28d/concept_audit.md) contient des
comptages historiques de faisabilité. Ils doivent être présentés avec leur date,
leur définition provisoire et leur environnement, jamais comme des résultats
actuels universels.

## Plan de secours pour une démonstration

- Tester la configuration et les services externes avant la séance.
- Conserver un `run_id` pré-généré pour une lecture hors ligne de `state.json`
  et `demo_report.md`.
- Utiliser `--skip-literature` ou `--skip-knowledge-graph` uniquement en
  annonçant clairement la réduction de preuve.
- Ne jamais afficher de secret, identifiant cloud privé ou ligne patient.
