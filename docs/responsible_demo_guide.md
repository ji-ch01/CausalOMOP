# Guide de démonstration aux responsables

Ce guide présente l'état réellement implémenté de CausalOMOP sans confondre
résultats techniques, résultats de faisabilité et résultats causaux.

## 1. Préparer le terminal

Depuis la racine du dépôt :

```powershell
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -v
python -m causal.cli config --json
python -m causal.cli preflight --count-persons
```

Résultats attendus : tests réussis, dataset OMOP configuré, dix tables OMOP
requises accessibles et 259 770 personnes.

## 2. Créer un run reproductible

```powershell
python -m causal.cli workflow start studies\albumin_sepsis_28d\protocol.json
```

Copier le `run_id` affiché. Chaque run conserve son propre protocole, son état,
ses décisions et ses artefacts sous `artifacts/runs/<run_id>/`.

## 3. Exécuter le Hypothesis Agent

```powershell
python -m causal.cli workflow hypothesis <run_id>
```

Montrer `hypothesis.json`. Il formalise population, traitement, comparateur,
outcome, time zero, estimands, confondeurs et problèmes non résolus. Souligner
que `executable: false` est une sécurité attendue et non une erreur.

## 4. Enregistrer la décision sur le protocole

Le reviewer doit utiliser son vrai nom ou identifiant :

```powershell
python -m causal.cli workflow gate <run_id> protocol approved `
  --reviewer "identifiant-reviewer" `
  --comment "Protocole publié retenu comme benchmark de développement"
python -m causal.cli workflow advance <run_id>
```

Cette décision valide le cas de développement, pas les mappings RxNorm.

## 5. Exécuter le Data Science Agent

```powershell
python -m causal.cli workflow plan <run_id>
```

Montrer `analysis_plan.json` : tables OMOP, phénotypes, modes standard ou
provisoire, covariables, fenêtres, contrôles de faisabilité et diagnostics.

## 6. Générer le support de réunion

```powershell
python -m causal.cli workflow demo-report <run_id>
Get-Content artifacts\runs\<run_id>\demo_report.md -Encoding UTF8
```

Le rapport affiche les gates, la question normalisée, le plan, les résultats de
l'audit et les limites. Il indique explicitement qu'aucun effet causal n'a encore
été estimé.

## 7. Résultats que l'on peut présenter

- environnement BigQuery et contrats opérationnels ;
- 259 770 personnes dans la table `person` ;
- concepts sepsis et choc septique trouvés et utilisés ;
- 4 838 personnes pour le concept sepsis générique `132797` ;
- 2 719 personnes pour le choc septique `196236` ;
- présence d'administrations d'albumine et de Lactated Ringer's ;
- détection automatique de leurs mappings standards manquants ;
- protocole et plan d'analyse reproductibles et traçables ;
- séparation effective entre exploration et analyse finale.
- premier comptage provisoire : 3 048 patients éligibles, 416 avec albumine et
  cristalloïdes, 2 496 avec cristalloïdes seuls ;
- écart avec les ~18 421 patients publiés, attendu tant que le phénotype
  diagnostic OMOP ne reproduit pas la définition clinique Sepsis-3.

## 8. Ce qu'il ne faut pas encore présenter comme résultat

- un effectif final de cohorte ;
- une différence de mortalité entre groupes ;
- un ATE, ATT ou CATE ;
- une conclusion clinique sur l'albumine ;
- une reproduction réussie de l'article publié.

Ces résultats nécessitent encore la définition d'ICU/time zero, la cohorte de
faisabilité, les covariables, les diagnostics et la validation des mappings.

## 9. Questions à soumettre aux responsables

1. Le protocole publié est-il accepté comme benchmark principal ?
2. L'usage temporaire des valeurs sources est-il accepté pour le développement ?
3. Qui valide les mappings RxNorm albumine et cristalloïdes ?
4. La population doit-elle reproduire Sepsis-3 ou utiliser les diagnostics OMOP ?
5. Quelles différences avec les effectifs publiés seront jugées acceptables ?
