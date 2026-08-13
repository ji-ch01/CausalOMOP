# Article figures

`figure_1_causalomop_workflow.svg` is an editable, vector-first replacement
for the hand-drawn all-in-one workflow. It is intended for a manuscript's
Methods/Architecture section and uses explicit SVG connector paths rather than
automatic Mermaid routing.

Suggested manuscript caption:

> **Figure 1. Governed CausalOMOP workflow and relationship to FastOMOP.**
> CausalOMOP separates causal study design, OMOP operationalization, and
> governed data execution. Descriptive questions are redirected to FastOMOP,
> which shares the OMCP data-access layer but cannot invoke the causal-query
> surface. LLMs and literature sources provide advisory inputs only; human
> reviewers approve the target trial, DAG/identification, clinical mapping,
> analysis plan, statistical estimand, and final results before an auditable
> report is produced. Patient-level analytic data are never sent to LLM
> providers.

For the manuscript, this figure should be complemented by two focused figures
rather than another end-to-end diagram:

1. A study-specific DAG (for example, albumin plus crystalloids versus
   crystalloids and 28-day mortality in sepsis), with observed and unobserved
   confounders distinguished.
2. A validation/results figure: cohort flow plus the prespecified effect
   estimate and diagnostics. It must be populated only after the corresponding
   approved run is available.

The diagram deliberately does not depict TMLE, CATE, IPCW, negative controls,
or E-value quantitative sensitivity analysis as separate boxes -- they are all
implemented (see [causal/README.md](../../causal/README.md#current-capabilities))
but folded into the single "governed data execution" stage for figure clarity.

FastOMOP is intentionally shown as a single external descriptive-workflow
component: its Semantic Agent and Database Agent are implementation detail for
the FastOMOP article, while FOEM is an offline SQL evaluation/monitoring tool,
not a CausalOMOP runtime step or a causal-validity check.
