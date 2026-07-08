# Local Data Directory

This folder is for local data only. Large database and vocabulary files are not
committed to Git.

Expected layout:

```text
data/
  README.md
  synthetic_snuh_raw.duckdb
  synthetic_snuh_curated.duckdb
  synthetic_snuh_v1.duckdb
  backups/
  vocabulary/
    CONCEPT.csv
    CONCEPT_RELATIONSHIP.csv
    CONCEPT_ANCESTOR.csv
    VOCABULARY.csv
    CONCEPT_SYNONYM.csv
    DRUG_STRENGTH.csv
```

Git should ignore:

- `*.duckdb`;
- `backups/`;
- `vocabulary/`;
- exported patient-level rows;
- intermediate analytic datasets that contain individual-level data.

The SNUH synthetic DuckDB files are development assets for OMOP/FastOMOP
infrastructure testing. They should not be treated as the main causal inference
validation dataset.

Use `docs/datasets.md` for dataset notes, rebuild instructions, and the public
dataset review plan.
