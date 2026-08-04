# SNUH ETL Scripts

These scripts prepare the local SNUH synthetic DuckDB database for OMOP
analysis and FastOMOP testing.

They are dataset-specific. Reusable causal logic should live under `causal/`,
not in this folder.

SYN-SNUH is a development dataset, not the primary causal-validation source.
For the active MIMIC-IV policy and local export path, see
[`docs/datasets.md`](../../docs/datasets.md). OMOP vocabulary downloads are
managed through [OHDSI Athena](https://athena.ohdsi.org/).

## Defaults

The scripts use:

```text
OMOP_DB_PATH=data/synthetic_snuh_curated.duckdb
OMOP_VOCAB_DIR=data/vocabulary
```

Override these variables when needed.

## Run Order

Run commands from the repository root:

```bash
python -m scripts.snuh_etl.load_vocabulary
python -m scripts.snuh_etl.fix_drugs
python -m scripts.snuh_etl.fix_measurements
python -m scripts.snuh_etl.remap_observations
python -m scripts.snuh_etl.extend_concept_ancestor
python -m scripts.snuh_etl.validate_vocabulary
```

Use `python -m ...` instead of executing the files by path. This keeps package
imports stable after the scripts were moved into `scripts/snuh_etl/`.

## Script Roles

| Script | Purpose |
|---|---|
| `etl_utils.py` | shared paths, database connection, and backup helpers |
| `load_vocabulary.py` | loads Athena vocabulary CSVs into DuckDB |
| `fix_drugs.py` | fixes drug concept-domain issues |
| `fix_measurements.py` | repairs known measurement mapping issues |
| `remap_observations.py` | remaps HSDN/MeSH observations to OMOP concepts |
| `remap_obs_umls.py` | auxiliary UMLS-oriented observation mapping work |
| `extend_concept_ancestor.py` | creates auditable local hierarchy extension/view |
| `validate_vocabulary.py` | reports CDM table and concept coverage |
| `add_temporal_cells.py` | notebook helper for adding temporal analysis cells |

## Backups

Write scripts create timestamped backups under:

```text
data/backups/
```

These backups are local artifacts and should not be committed.
