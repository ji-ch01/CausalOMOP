# Notebooks

This folder contains exploratory analysis notebooks.

| Notebook | Purpose |
|---|---|
| `omop_analysis.ipynb` | general OMOP/SNUH readiness analysis |
| `t2dm_analysis.ipynb` | Type 2 Diabetes treatment, measurement, and adverse-event exploration |
| `depressive_disorder_analysis.ipynb` | depressive disorder exploratory analysis |

## Policy

Use notebooks for:

- exploration;
- figures;
- data-quality investigation;
- research notes.

Move reusable logic into:

```text
scripts/snuh_etl/   for SNUH-specific ETL
causal/             for reusable causal workflow code
```

Avoid committing large rendered outputs or local data exports.

