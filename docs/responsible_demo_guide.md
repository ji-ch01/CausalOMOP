# Responsible Demonstration Guide (English)

Present CausalOMOP as a governed study-design and feasibility prototype, not as
a causal-effect engine or clinical decision system. This guide uses the
albumin/sepsis study as its running example because it is the pipeline's
worked literature-comparison case; make clear to any audience that CausalOMOP
itself is general-purpose and disease-agnostic, and that the study is a
validation example, not the tool's scope (see
[README.md's References](../README.md#references)).

## Opening statement

> CausalOMOP turns a clinical question into an auditable target-trial draft,
> evidence-grounded candidate DAG, identification review, OMOP mappings, and
> feasibility diagnostics. LLM outputs remain proposals, important decisions
> require human review, and causal estimation is not implemented yet.

## Setup

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest tests -q
python -m causal.cli config --json
```

With BigQuery and Application Default Credentials configured:

```powershell
python -m causal.cli preflight --count-persons
```

Show the live output. Do not quote a frozen test count, table count, or person
count from documentation.

## Create a governed run

From the versioned study protocol:

```powershell
$state = python -m causal.cli workflow start studies\albumin_sepsis_28d\protocol.json |
  ConvertFrom-Json
$runId = $state.run_id
python -m causal.cli workflow hypothesis $runId
```

Or from a natural-language question:

```powershell
python -m causal.cli workflow start-question "Among adult ICU patients with sepsis, what is the effect of albumin plus crystalloids versus crystalloids alone on 28-day mortality?"
```

Explain that the first LLM step classifies the question. Descriptive requests
are redirected to FastOMOP; ambiguous requests return a clarification question.
The causal path can retrieve PubMed/OpenAlex evidence, use OptimusKG when
available, and run a bounded DAG critique/revision loop.

## Show the audit trail

Under `artifacts/runs/<run_id>/`, show:

- `protocol.json` for the target-trial draft;
- `hypothesis.json` for the candidate DAG and identification result;
- `llm_outputs.json` for raw drafting, critique, and revision responses;
- `state.json` for phases, gate decisions, reviewers, and history.

An `executable: false` value is an expected safety state while reviews remain
incomplete.

Export the DAG for independent inspection:

```powershell
python -m causal.cli workflow dag-export <run_id> --highlight-adjustment-set
```

The output can be pasted into [dagitty.net](https://dagitty.net). A graph that
passes a backdoor test is not automatically clinically complete or correct.

## Human review and OMOP mapping

```powershell
python -m causal.cli workflow gate <run_id> protocol approved `
  --reviewer "reviewer-id" `
  --comment "Target trial and DAG accepted for technical exploration"
python -m causal.cli workflow advance <run_id>
python -m causal.cli workflow plan <run_id> --with-mcp
```

OMCP lookup results are candidates only. Review mappings through
`concept-set-map`, the `covariate-*` commands, or the Streamlit interface:

```powershell
streamlit run causal/ui/streamlit_app.py
```

Never approve a mapping or gate merely to make a live demo proceed.

## Registry and aggregate diagnostics

After the required reviews:

```powershell
python -m causal.cli workflow registry-build <run_id>
python -m causal.cli workflow gate <run_id> clinical_mapping provisional `
  --reviewer "reviewer-id" --comment "Aggregate audits only"
python -m causal.cli workflow advance <run_id>
python -m causal.cli workflow feasibility <run_id> --transport mcp
python -m causal.cli workflow execution-readiness <run_id>
python -m causal.cli workflow cohort-plan <run_id>
python -m causal.cli workflow cohort-profile <run_id>
python -m causal.cli workflow followup-audit <run_id>
python -m causal.cli workflow mapping-audit <run_id> <variable_id>
python -m causal.cli workflow demo-report <run_id>
```

Commands correctly fail closed when run state, mappings, or gates are
incomplete. Demonstrating that refusal is part of the governance story.

## Safe claims

- The system creates versioned, reviewable protocol and DAG drafts.
- Retrieved evidence identifiers are checked against what was actually fetched.
- Human decisions and execution blockers are persisted.
- OMOP mappings and aggregate coverage can be audited.
- Aggregate feasibility results can be reported with their date, query
  definition, and environment.
- The system can generate governed analytic-cohort SQL.

## Claims to avoid

- A provisional count is a final cohort.
- The generated DAG is clinically authoritative.
- A mortality difference is a treatment effect.
- CausalOMOP has estimated an ATE, ATT, or CATE.
- The albumin study has been reproduced.
- The system is ready for clinical use.

Historical counts in the study's
[concept audit](../studies/albumin_sepsis_28d/concept_audit.md) are dated
feasibility snapshots, not portable current results.

## Live-demo fallback

- Test configuration and external services in advance.
- Keep a pre-generated run for offline inspection.
- Use `--skip-literature` or `--skip-knowledge-graph` only while explicitly
  stating the reduced evidence path.
- Never display secrets, private cloud identifiers, or patient-level rows.
