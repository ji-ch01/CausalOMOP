# Notebooks

These notebooks are exploratory research artifacts, not workflow entry points
or production implementations.

| Notebook | Purpose |
|---|---|
| `omop_analysis.ipynb` | general OMOP and historical SYN-SNUH readiness exploration |
| `t2dm_analysis.ipynb` | type 2 diabetes treatment and outcome exploration |
| `depressive_disorder_analysis.ipynb` | depressive-disorder exploration |

Notebook findings are snapshots tied to their input data and execution state.
They must not be treated as current MIMIC-IV validation results unless the
notebook records that dataset, configuration, date, and reproducible run.

Use notebooks for investigation, figures, and research notes. Move reusable
SYN-SNUH preparation to `scripts/snuh_etl/` and reusable workflow logic to
`causal/`. Do not commit embedded patient-level rows, credentials, database
files, large rendered outputs, or local paths.

See [docs/datasets.md](../docs/datasets.md) for the current dataset policy.
