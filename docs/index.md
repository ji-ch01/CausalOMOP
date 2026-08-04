# Documentation Index

This page is the entry point for CausalOMOP documentation. It distinguishes
maintained specifications from historical notes and generated artifacts.

## Maintained documentation

| Document | Purpose |
|---|---|
| [Project README](../README.md) | concise setup, workflow, and project boundary |
| [Causal package](../causal/README.md) | implemented modules, trust boundaries, and limitations |
| [Causal roadmap](causal_roadmap.md) | normative scientific order and current implementation gap |
| [Architecture diagrams](project_architecture_diagrams.md) | current end-to-end and gate diagrams |
| [Datasets](datasets.md) | MIMIC-IV, DuckDB export, SYN-SNUH, and data policy |
| [FastOMOP integration](fastomop_integration.md) | component ownership, OMCP contract, and submodules |
| [LLM hypothesis integration](llm_hypothesis_integration.md) | question gate, evidence retrieval, critique, and provider policy |
| [DAG audit](dag_generation_audit.md) | dated technical audit and metric history |
| [Repository map](repo_map.md) | folder ownership and Git policy |
| [Responsible demo guide](responsible_demo_guide.md) | current French demonstration script |
| [Responsible demo guide (English)](responsible_demo_guide_en.md) | English presentation companion |
| [Albumin/sepsis study](../studies/albumin_sepsis_28d/README.md) | first literature-backed validation case |

## Historical records

These files preserve design history but are not implementation references:

- [original design proposal](../document_design.md);
- [earlier causal architecture](causal_architecture.md);
- [dataset-selection work package](reproducible_causal_review.md).

If a historical document conflicts with a maintained page, the maintained page
and the executable CLI help are authoritative.

## External documentation

CausalOMOP does not duplicate documentation owned by its submodules or data
providers:

- [FastOMOP](https://github.com/ji-ch01/agno_fastomop);
- [OMCP](https://github.com/ji-ch01/omcp);
- [FOEM](https://github.com/ji-ch01/foem);
- [OHDSI MIMIC-to-OMOP ETL](https://github.com/OHDSI/MIMIC);
- [OMOP Common Data Model](https://github.com/OHDSI/CommonDataModel);
- [MIMIC-IV on PhysioNet](https://physionet.org/content/mimiciv/2.2/);
- [OHDSI Athena vocabularies](https://athena.ohdsi.org/).

## Excluded from documentation maintenance

Files under `causal/prompts/` are runtime prompt templates, not prose
documentation. Files under `artifacts/runs/` are generated snapshots. Neither
should be edited merely to synchronize documentation; regenerate run artifacts
through the workflow when a new demonstration snapshot is required.
