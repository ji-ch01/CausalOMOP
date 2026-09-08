# Documentation Index

This page is the entry point for CausalOMOP documentation. It distinguishes
maintained specifications from historical notes and generated artifacts.

CausalOMOP is a general-purpose, protocol-driven causal-inference tool, not a
system built around one disease or drug. Where a document below discusses a
specific clinical question (PPI vs H2RA stress-ulcer prophylaxis, a MIMIC-IV
table's vocabulary coverage, ...), that question is a worked example used to validate and
compare pipeline output against published literature -- see
[README.md's References](../README.md#references) -- not a description of
the tool's scope.

## Maintained documentation

| Document | Purpose |
|---|---|
| [Project README](../README.md) | concise setup, workflow, and project boundary |
| [Causal package](../causal/README.md) | implemented modules, trust boundaries, and limitations |
| [Causal roadmap](causal_roadmap.md) | normative scientific order and current implementation gap |
| [Architecture diagrams](project_architecture_diagrams.md) | current end-to-end and gate diagrams |
| [Datasets](datasets.md) | MIMIC-IV, DuckDB export, SYN-SNUH, and data policy |
| [Vocabulary mapping coverage](vocabulary_mapping_coverage.md) | source-data limitation note: OMOP concept mapping completeness, quantified over time |
| [FastOMOP integration](fastomop_integration.md) | component ownership, OMCP contract, and submodules |
| [LLM hypothesis integration](llm_hypothesis_integration.md) | question gate, evidence retrieval, critique, and provider policy |
| [DAG generation reference](dag_generation_audit.md) | pipeline design, evaluation methodology (SHD/SID, causal-learn), and known gaps |
| [Eligibility refinements design](eligibility_refinements_design.md) | washout / new-user (implemented), landmark and in-hospital-mortality modes (specified); the methodological corrections each needs |
| [Repository map](repo_map.md) | folder ownership and Git policy |
| [Responsible demo guide](responsible_demo_guide.md) | current demonstration script |
| [Validation study candidates](validation_study_candidates.md) | reference-grounded shortlist of causal questions runnable end-to-end, with selection criteria and citations |
| [Validation question bank](validation_question_bank.md) | ten ETL-feasible causal questions, each specified as a target trial with a published reference truth, for the multi-case evaluation |
| [PPI vs H2RA study](../studies/ppi_vs_h2ra_sup_mortality/README.md) | question-bank Q1: active-comparator emulation anchored to PEPTIC; scaffolded, pending review |
| [Corticosteroids in severe CAP study](../studies/corticosteroids_severe_cap_mortality/README.md) | question-bank Q8: sensitivity case anchored to CAPE COD; scaffolded, pending review |
| [Vancomycin + pip-tazo vs cefepime AKI study](../studies/vanco_piptazo_vs_cefepime_aki/README.md) | question-bank Q5: headline diagnostic case anchored to ACORN; scaffolded, pending review |

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
