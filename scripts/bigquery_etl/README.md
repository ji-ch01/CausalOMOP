# BigQuery-to-DuckDB Export

This standalone utility copies the configured BigQuery OMOP dataset into a
local DuckDB file for lower-latency development. It does not import the
`causal` package and does not change source tables.

From the repository root:

```powershell
pip install -r requirements-bigquery.txt
python -m scripts.bigquery_etl.export_duckdb data/mimiciv_omop.duckdb
```

The exporter discovers all tables by default, streams Arrow batches to keep
memory bounded, and compares source and destination row counts. Use:

```powershell
python -m scripts.bigquery_etl.export_duckdb data/mimiciv_omop.duckdb `
  --tables person concept --schema main
```

Run `python -m scripts.bigquery_etl.export_duckdb --help` for all options,
including `--overwrite` and `--no-verify`.

The output contains the same governed clinical data as the BigQuery source.
Keep it access-controlled, never commit it, and never place it in a public or
shared location. Configure CausalOMOP to use it as described in
[docs/datasets.md](../../docs/datasets.md).
