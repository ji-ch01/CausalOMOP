# Datasets and Data Policy

CausalOMOP separates source-data access, OMOP conversion, local development
copies, generated run artifacts, and causal claims. No patient-level dataset is
distributed with the repository.

## Primary research dataset

The current validation environment uses credentialed
[MIMIC-IV 2.2](https://physionet.org/content/mimiciv/2.2/) converted to OMOP CDM
5.3.1 using the [OHDSI/MIMIC](https://github.com/OHDSI/MIMIC) project. The
conversion is queried either in BigQuery or from a controlled local DuckDB
export.

MIMIC-IV access remains subject to PhysioNet credentialing and the applicable
data-use agreement. Conversion does not make the data public. Local exports,
query results, run artifacts containing sensitive context, and derived
person-level datasets inherit the source restrictions.

## Runtime backends

Set `CAUSALOMOP_DB_BACKEND` to `duckdb` (recommended) or `bigquery`.

`sample.env` defaults new setups to `duckdb`. The hard-coded default inside
`BigQueryDatasets` (`causal/config.py`) stays `bigquery` on purpose, so a
missing/misplaced `.env` fails loudly against a placeholder project (see
`FromEnvPlaceholderGuardTests` in `tests/test_config.py`) instead of silently
resolving to a possibly-stale local file, and so BigQuery stays reachable with
one flag -- no code change -- if the local export is ever unavailable. This
mirrors `requirements-bigquery.txt` already being an optional install rather
than a base dependency.

### DuckDB (recommended)

```text
CAUSALOMOP_DB_BACKEND=duckdb
CAUSALOMOP_DUCKDB_PATH=data/mimiciv_omop.duckdb
CAUSALOMOP_DUCKDB_SCHEMA=main
```

Queries run against a local file created by
[BigQuery-to-DuckDB export](#bigquery-to-duckdb-export) below. Preferred for
day-to-day development and demos: access is controlled directly (filesystem
permissions on a file you hold), there is no persistent copy of the export
sitting in a third-party cloud project beyond the source BigQuery dataset
itself, and query latency is much lower. The export must be refreshed
whenever the source BigQuery dataset changes -- a stale local file silently
diverges from BigQuery with no built-in warning, so treat its export date as
part of any result you report.

### BigQuery (secondary)

```text
CAUSALOMOP_DB_BACKEND=bigquery
```

| Role | Environment variable |
|---|---|
| GCP project | `CAUSALOMOP_BQ_PROJECT` |
| analysis-ready OMOP CDM | `CAUSALOMOP_BQ_OMOP_DATASET` |
| vocabulary preparation/audit | `CAUSALOMOP_BQ_VOCAB_DATASET` |
| staging and intermediate ETL | `CAUSALOMOP_BQ_ETL_DATASET` |
| tests, run logs, and metrics | `CAUSALOMOP_BQ_METRICS_DATASET` |

Clinical analyses must read from the configured OMOP role, not silently from
staging tables. The analysis-ready OMOP dataset is treated as read-only. Use
this backend when no local DuckDB export exists yet, or to query the live
source directly (e.g. to confirm a result isn't an artifact of a stale
export).

The CausalOMOP configuration above is distinct from the environment passed to
the OMCP subprocess (`DB_TYPE`, `DB_PATH`, `CDM_SCHEMA`, `VOCAB_SCHEMA`). The
CausalOMOP MCP client resolves and supplies the appropriate OMCP values.

Check the resolved configuration without querying a database:

```powershell
python -m causal.cli config --json
```

## BigQuery-to-DuckDB export

The standalone exporter copies the configured BigQuery OMOP dataset into one
local DuckDB schema using Arrow batches and verifies source/destination row
counts by default:

```powershell
pip install -r requirements-bigquery.txt
python -m scripts.bigquery_etl.export_duckdb data/mimiciv_omop.duckdb
```

Useful options:

```powershell
python -m scripts.bigquery_etl.export_duckdb data/mimiciv_omop.duckdb `
  --tables person concept --schema main
```

- `--tables` exports a subset;
- `--schema` chooses the destination schema;
- `--overwrite` replaces destination tables that already exist;
- `--no-verify` skips row-count verification.

See [scripts/bigquery_etl/README.md](../scripts/bigquery_etl/README.md). The
output file must remain ignored, access-controlled, and outside public or shared
locations.

## SYN-SNUH

The synthetic SNUH DuckDB assets remain useful for:

- ETL and vocabulary tests;
- descriptive FastOMOP smoke tests;
- OMOP concept-coverage checks;
- early workflow and SQL mechanics.

They are not the primary causal-validation dataset. Known limitations include
synthetic temporal artifacts, incomplete or insufficiently detailed baseline
variables, sparse vital signs, limited adverse-event signal, some prevalent-user
patterns, and incomplete vocabulary hierarchy coverage.

Prepare SYN-SNUH using [scripts/snuh_etl/README.md](../scripts/snuh_etl/README.md).

## Local layout

```text
data/
  README.md
  mimiciv_omop.duckdb              # restricted local export
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

All database files, WAL files, backups, vocabulary exports, and derived
person-level datasets must stay out of Git.

Vocabulary downloads are managed through
[OHDSI Athena](https://athena.ohdsi.org/) and remain subject to their source
licences.

## Validation study

Dataset selection is complete. CausalOMOP itself is not scoped to any one
condition; the active work is running the pipeline end-to-end on one
literature-comparison case -- the albumin/sepsis target trial -- against the
converted MIMIC-IV data, so its output can be checked against that published
study's design and results. The study still requires a reproducible Sepsis-3
phenotype, approved albumin and crystalloid mappings, validated ICU time zero,
baseline covariate coverage, censoring and follow-up rules, and positivity
assessment.

See [studies/albumin_sepsis_28d](../studies/albumin_sepsis_28d/README.md), the
[causal roadmap](causal_roadmap.md)'s Validation strategy, and
[README.md's References](../README.md#references) for the full set of
literature this pipeline is grounded in and compared against.

## Handling rules

- Never commit credentials, project identifiers, database files, raw query
  results, or patient-level analytic tables.
- Never send patient-level data or sensitive free text to an LLM provider.
- Treat `artifacts/runs/` as research records that may contain submitted
  clinical questions and review metadata.
- Prefer aggregate diagnostics for review and demonstrations.
- A generated SQL plan is not authorization to export, publish, or share its
  result.
- Follow the source data-use agreement, institutional policy, and reviewer
  decisions even when a technical control would permit more.
