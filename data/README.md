# Local Data Directory

This directory is a mount point for local data. Only this README belongs in
Git; databases, WAL files, vocabularies, backups, and derived datasets do not.

```text
data/
  README.md
  mimiciv_omop.duckdb              # restricted MIMIC-IV OMOP export
  synthetic_snuh_raw.duckdb
  synthetic_snuh_curated.duckdb
  synthetic_snuh_v1.duckdb         # optional historical copy
  backups/
  vocabulary/
    CONCEPT.csv
    CONCEPT_RELATIONSHIP.csv
    CONCEPT_ANCESTOR.csv
    VOCABULARY.csv
    CONCEPT_SYNONYM.csv
    DRUG_STRENGTH.csv
```

MIMIC-IV exports retain the source dataset's credentialing and data-use
restrictions. SYN-SNUH files are development assets for ETL, vocabulary, and
FastOMOP smoke tests; they are not the primary causal-evidence source.

See [docs/datasets.md](../docs/datasets.md) for backend configuration, export
instructions, access rules, and the active validation study.
