# FastOMOP Integration

This project uses OMCP as the shared governed OMOP access layer for both
FastOMOP descriptive queries and CausalOMOP causal workflows.

Component-specific installation and API details remain authoritative in the
official repositories: [FastOMOP](https://github.com/ji-ch01/agno_fastomop),
[OMCP](https://github.com/ji-ch01/omcp), and
[FOEM](https://github.com/ji-ch01/foem). This page documents only the
CausalOMOP integration and governance boundary.

## Division of Responsibility

| Component | Role | Never does |
|---|---|---|
| FastOMOP (`agno_fastomop/`) | Semantic exploration and descriptive NL-to-SQL over OMOP data | Never defines or approves a causal variable mapping; never produces the final causal cohort/covariate SQL |
| OMCP (`omcp/`) | Governed concept search (`Lookup_Drug`/`Lookup_Condition`/`Lookup_Concept`), concept metadata (`Get_Concepts_By_Id`), aggregate audits, and governed execution (`Causal_Select_Query`) | Never approves a clinical mapping -- every concept it returns is a *candidate*, not an approved concept (see `causal.mapping.registry.OMOPVariableDefinition.candidate_concept_ids`) |
| CausalOMOP (`causal/`) | Causal-question/DAG definition, human mapping approval, the OMOP variable registry (`causal.mapping.registry`), and deterministic SQL compilation from an approved registry entry | Never lets FastOMOP generate the final causal SQL directly; never treats an OMOP-lookup candidate as approved |
| FOEM (`foem/`) | SQL regression/benchmark testing for FastOMOP's descriptive queries | Not a causal-validity check -- passing FOEM says nothing about whether a causal mapping or DAG is correct |

The OMOP variable registry (`causal.mapping.registry.OMOPVariableRegistry`,
`causal.mapping.audit`) is what makes this division concrete for mapping
work specifically: FastOMOP/OMCP tools may only ever populate a registry
entry's `candidate_concept_ids` (via `causal.agents.data_science`'s
`ConceptLookup`); only a human reviewer, through the existing
`causal.orchestrator.workflow` gates (`review_covariate`,
`map_covariate_concepts`, `map_concept_set`, ...), can move a concept into
`root_concept_ids` with `mapping_status = "approved"` -- nothing in
`causal.mapping` itself ever does this automatically. See
[causal roadmap](causal_roadmap.md) for the full design.

## Included FastOMOP Repositories

| Path | Upstream role | Used for |
|---|---|---|
| `agno_fastomop/` | FastOMOP implementation using Agno agents | NL-to-SQL clinical query workflow |
| `omcp/` | Model Context Protocol server for OMOP CDM | database tools exposed to agents |
| `foem/` | FastOMOP Evaluation and Monitoring | benchmark SQL query evaluation |

These are submodules. They are intentionally kept separate from project-owned
causal and ETL code.

## Runtime Relationship

```text
User query
  |
  v
agno_fastomop
  |
  +-- Semantic Agent maps clinical terms to OMOP concepts
  |
  +-- Database Agent generates and executes SQL
          |
          v
        omcp
          |
          v
        OMOP database
```

`foem` is not required for normal runtime. It is useful for evaluation.

## Relationship to NL-to-SQL

Causal questions still need:

- concept lookup;
- exposure/comparator mapping;
- outcome concept mapping;
- cohort SQL;
- covariate SQL;
- database execution.

For that reason, the causal extension should not be a completely separate
top-level agent team that bypasses NL-to-SQL. A practical architecture is:

```text
FastOMOP orchestrator
  |
  v
NL-to-SQL team
  |
  +-- descriptive mode
  |     semantic context -> SQL answer
  |
  +-- causal mode
        causal study spec -> OMOP cohort/covariate SQL -> estimate/report
```

The current CausalOMOP CLI and UI classify a question before drafting. A
descriptive question is redirected to FastOMOP, but the applications are not
yet connected by automatic process-level routing. A causal question must create
a target-trial specification before SQL is generated.

## Shared MCP Contract

FastOMOP remains backward compatible with the existing `Select_Query` tool.
CausalOMOP uses `Causal_Select_Query`, which additionally requires:

- an operation (`aggregate_feasibility`, `cohort_build`, `diagnostics`, or
  `causal_estimation`);
- the workflow `run_id` and `study_id`;
- the workflow mode and gate decisions;
- the boolean permissions produced by the Execution Agent.

Each readiness evaluation writes `mcp_context.json` next to the run artifacts.
OMCP validates this context and fails closed before submitting unauthorized
causal SQL. This is a technical authorization check and does not replace human
clinical validation.

For the MIMIC-IV OMOP conversion, configure the shared server with:

```text
DB_TYPE=bigquery
DB_PROJECT=<your-gcp-project>
DB_DATASET=<your-omop-dataset>
DB_READ_ONLY=true
CDM_SCHEMA=<your-omop-dataset>
VOCAB_SCHEMA=<your-omop-dataset>
```

Google Application Default Credentials are used for authentication.

Install the CausalOMOP MCP client and run governed feasibility with:

```powershell
pip install -r requirements-mcp.txt
python -m causal.cli workflow feasibility <run_id> --transport mcp
```

The Data Science Agent can also retrieve standardized vocabulary candidates:

```powershell
python -m causal.cli workflow plan <run_id> --with-mcp
```

These candidates are written under each phenotype in `analysis_plan.json`. They
never change `concept_ids`, approve a mapping, or enable execution; a reviewer
must copy approved identifiers into the protocol through the existing clinical
mapping gate.

`Lookup_Drug`/`Lookup_Condition`/`Lookup_Concept` match a search term against
either a concept's own name or any of its `concept_synonym` entries
(abbreviations, alternate names), and accept `limit`/`offset` for paging past
the first page of results; `Lookup_Concept` also orders results by
`vocabulary_id` first so a mixed-vocabulary result set (e.g. LOINC and SNOMED
both matching a measurement) groups by vocabulary instead of interleaving.

A non-demographic covariate is only actually **extracted** once both reviews
are done: `workflow covariate-map-concepts` (`mapping_status="approved"`,
real `concept_ids`) and `workflow covariate-review-aggregation` (how to
combine repeated occurrences within its window -- presence/count for
condition/drug/procedure, or a `value_as_number` aggregate for measurement/
observation). Only then does the covariate's `status` become `"ready"`, and
`workflow cohort-plan` adds it as a real column to the generated cohort SQL
(see `causal.data_builder.analytic.AnalyticCohortBuilder`,
`causal.queries.feasibility.CovariateSpec`). The OMCP execution path itself
is unchanged: `Causal_Select_Query` still runs whatever SQL
`AnalyticCohortBuilder` produces, now including these covariate columns when
present.

`workflow covariate-map-concepts` also accepts `--with-mcp`: when passed, it
calls the `Get_Concepts_By_Id` tool to verify each approved `concept_id`'s
real OMOP `domain_id` actually matches the covariate's declared domain,
rejecting the mapping otherwise (a reviewer can type a concept_id directly
rather than picking a search result, so nothing else confirms it). On the
CLI this stays opt-in; the Streamlit review UI runs it unconditionally in
its "Save concept mapping" form, since that app is where a reviewer with
real BigQuery/OMOP access does the actual clinical mapping review.

`workflow concept-set-map` is the equivalent governed action for
`concept_sets` -- the domain + concept_ids a protocol declares for
population/treatment/comparator/outcome (and any other declared phenotype),
which both `DataScienceAgent` (phenotype review) and SQL generation
(`_load_concept_specs`, feeding `feasibility_sql`/`AnalyticCohortBuilder`)
read from this single store, by role. There used to be a second,
independently-mapped `query_concept_sets` store for the same role -- removed,
since nothing checked the two agreed, and mapping the same concept twice was
pure duplicate review work. `concept-set-map` accepts `--with-mcp` for a
domain cross-check. The Streamlit review UI has an editable form for this too
(live `Search OMCP` + approve concept_id(s), one per `concept_sets` entry),
so this is no longer CLI/manual-`protocol.json`-edit only.

The resulting `feasibility.json` records `execution_transport = "omcp"`, the
OMCP query hash, run identifier, study identifier, and causal permission flag.

`workflow mapping-audit <run_id> <variable_id>` runs `causal.mapping.audit`'s
aggregate-only SQL diagnostics for one `registry.json` variable (any approved
phenotype -- population/treatment/comparator/outcome -- or covariate) through
the same governed OMCP path, and writes `mapping_audit_<variable_id>.json`.
Every concept-based-domain variable gets: row/person counts per included
concept, approved root concepts missing from the data, invalid/non-standard
concepts, and three cohort-entry diagnostics adapted from OHDSI's
CohortDiagnostics methodology (Rao et al. 2025, PLOS ONE,
doi:10.1371/journal.pone.0310634; see `causal/mapping/audit.py`'s module
docstring for the two simplifications made) -- `orphan_concepts` (related
concepts present in the data but not in the approved set), `index_event_breakdown`
(which specific concept drove each person's earliest qualifying event), and
`incidence_rate` (crude, unstratified entries per 1,000 person-years).
Measurement/observation variables also get `observed_units`. Diagnostic only:
this never changes `mapping_status` or blocks extraction -- a reviewer acts on
a finding through the existing `covariate-map-concepts`/`concept-set-map`
review commands.

## Governance Enforcement Boundary

Two things worth being explicit about, since a security audit (2026-08-04)
found the first genuinely broken and asked for a decision on the second.

**FastOMOP cannot reach `Causal_Select_Query`.** `agno_fastomop`'s workflow
(`workflows/omop_workflow.py`) builds one shared `MCPTools` connection for
both of its agents. That connection is scoped with an explicit `include_tools`
allowlist (`Get_Information_Schema`, `Select_Query`, `Lookup_Drug`,
`Lookup_Condition`, `Lookup_Concept`, `Get_Concepts_By_Id`, `Get_Capabilities`)
that does not include `Causal_Select_Query`. This matters because
`causal_context` (see above) is a plain dict supplied by whichever caller
invokes the tool -- OMCP validates its *shape* and its boolean gates, but has
no session binding back to a real CausalOMOP orchestrator run, so a
network-facing LLM agent with access to the tool could otherwise fabricate an
"approved" readiness snapshot. The allowlist removes that tool from FastOMOP's
reachable surface entirely, rather than relying on the agent's prompt to
never ask for it.

**The three OMCP-diagnostic paths in `causal.orchestrator.workflow`
(`run_cohort_profile_mcp`, `run_followup_audit_mcp`, `run_mapping_audit_mcp`)
send builder-generated SQL to `Causal_Select_Query` with no local
`QueryValidationAgent` pre-check, and this is intentional, not an oversight.**
`QueryValidationAgent`'s final-projection check (`causal/agents/
query_validation.py`) is a word-boundary regex, not a SQL parser -- it cannot
tell `COUNT(DISTINCT t.person_id)` (a legitimate aggregate, fine for
`aggregate_feasibility`) apart from a bare `person_id` projection (a real
patient-level leak). Applying it to these three paths would produce false
positives on the exact aggregate SQL `causal.mapping.audit` and
`AnalyticCohortBuilder.profile_sql`/`followup_audit_sql` are built to emit.
The decision (audit finding F-03/F-04, confirmed 2026-08-04): keep it that
way. `omcp/src/omcp/sql_validator.py` (single-statement, OMOP-table-only,
source-value-column-blocking) and `omcp/src/omcp/governance.py`
(`CausalExecutionContext`/`authorize_causal_operation`, fail-closed on any
missing or unapproved gate) are therefore the **sole** enforcement point for
these three paths. Any future change to either file is security-critical and
must ship with tests, the same way the F-01 stacked-statement fix
(`sg.parse()` + single-statement check in `validate_sql`) did.

## Local Configuration

Key files:

| File | Purpose |
|---|---|
| `agno_fastomop/config.toml` | model/provider and agent settings |
| `agno_fastomop/.env` | local LLM and tracing environment |
| `omcp/sample.env` | OMCP environment example |
| `docker-compose.yml` | integrated local runtime |

Check that the configured database path points to the file that actually exists.
For example, if the mounted file is:

```text
data/synthetic_snuh_curated.duckdb
```

then the container path should be:

```text
DB_PATH=/app/data/synthetic_snuh_curated.duckdb
```

## Docker Run

From the repository root:

```bash
docker compose up --build
```

Before running, verify:

- `OPENAI_API_KEY` or the chosen provider key is available;
- `DB_PATH` matches the mounted DuckDB filename;
- `DB_HOST_PATH` points to the local folder containing the database.

## Local Run

Install and run FastOMOP from the submodule:

```bash
cd agno_fastomop
uv sync
uv run python -m agno_fastomop.run_agent
```

`agno_fastomop` launches `omcp` through its MCP command configuration.

## Evaluation

`foem` contains benchmark queries:

```bash
cd foem
uv run python src/foem/sql_test.py
```

Use this for NL-to-SQL regression checks, not for causal validity.

## Submodule Hygiene

Inspect submodule status regularly:

```bash
git submodule status --recursive
git -C agno_fastomop status -sb
git -C omcp status -sb
git -C foem status -sb
```

If a submodule is edited, commit inside that submodule first, then commit the
updated pointer in the root repo.
