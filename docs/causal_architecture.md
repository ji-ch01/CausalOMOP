# Causal Architecture — Historical Note

This page documented the early SYN-SNUH and standalone cohort-builder design.
That architecture has been superseded.

Current references:

- [implementation overview](../causal/README.md);
- [normative causal roadmap](causal_roadmap.md);
- [current architecture diagrams](project_architecture_diagrams.md);
- [FastOMOP/OMCP integration](fastomop_integration.md).

The current system includes evidence-grounded DAG drafting, a causal-question
gate, deterministic identification checks, human review actions, an OMOP
variable registry, governed feasibility/audit execution, and analytic-cohort SQL
generation. Causal estimation, sensitivity analyses, and a final causal report
are still not implemented.

This file is retained so old links continue to resolve. Git history contains
the original design text.
