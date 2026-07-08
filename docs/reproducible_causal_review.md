# Reproducible Causal Study Review

This work package identifies a public validation dataset and a causal question
that can be reproduced or approximated with the CausalOMOP workflow.

## Purpose

The project should not validate causal reasoning only on the SYN-SNUH synthetic
dataset. SYN-SNUH remains useful for infrastructure, ETL, OMOP vocabulary
handling, FastOMOP testing, and early cohort-builder prototyping, but it may not
contain enough clinically realistic temporal structure, outcome signal, or
covariate coverage to validate a causal inference pipeline. In particular,
important baseline variables such as BMI or related body-size information may be
absent or insufficiently detailed for cardiometabolic causal questions.

The next validation target should be a public dataset, ideally one that has
already been used in a published causal study. This lets the pipeline output be
compared with published target trial choices, variables, assumptions, and
results.

## Review Questions

- Which reproducible causal studies use public or accessible clinical datasets?
- Which datasets contain exposure, outcome, follow-up, and baseline covariates
  required for target trial emulation?
- Which datasets are already OMOP CDM, can be converted to OMOP, or can be
  queried through an OMOP-compatible layer?
- Which causal questions support CATE estimation rather than only average
  treatment-effect estimation?
- Which published results can serve as future validation targets?

## Dataset Screening Criteria

For each candidate dataset, record:

- access conditions and data-use restrictions;
- OMOP compatibility or required ETL work;
- exposure coding and timing quality;
- outcome coding and event counts;
- baseline history before treatment or intervention;
- candidate confounders, mediators, and colliders;
- candidate effect modifiers for CATE;
- missingness, follow-up, censoring, and death handling;
- sample size and overlap/positivity expectations;
- relevant published causal studies.

## Causal Question Screening Criteria

For each candidate causal question, record:

- target population;
- treatment or intervention;
- comparator or control strategy;
- outcome and follow-up window;
- target trial design;
- primary estimand, with CATE preferred when feasible;
- clinically meaningful effect modifiers;
- variables required from the dataset;
- published estimates or expected qualitative findings;
- limitations when approximating the published design.

## Methodological Target

The primary causal objective is inference of conditional average treatment
effects (CATE) using CDM data. The planned estimator family should include:

- TMLE with SuperLearner-style nuisance modeling;
- g-computation;
- secondary ATE or ATT summaries when needed for comparison with the source
  study;
- diagnostics for positivity, overlap, covariate balance, missingness, and
  sensitivity to outcome/censoring definitions.

## Expected Deliverables

- candidate dataset table;
- candidate causal question table;
- variable availability matrix;
- OMOP compatibility assessment;
- source literature summary;
- recommended validation question;
- expected result benchmark for future pipeline validation.
