# CausalOMOP Causal Roadmap

## Product objective

CausalOMOP must answer a causal question from longitudinal observational data
stored in OMOP CDM. It is not a DAG generator and it is not a natural-language
wrapper around an estimator. Its output is an auditable causal study whose
scientific design exists before database extraction or model fitting.

The architecture follows the Causal Roadmap and target-trial emulation described
as grounding frameworks for medical causal co-pilots in Alaa et al. (2024).

## Required order of operations

```text
Natural-language causal question
  -> 1. Clarify the causal question and clinical equipoise
  -> 2. Specify the target trial
  -> 3. Define the counterfactual causal estimand
  -> 4. Construct and review the causal DAG
  -> 5. Identify the effect and an admissible adjustment set
  -> 6. Map every required variable to OMOP concepts, tables and time windows
  -> 7. Audit data support, missingness, positivity and follow-up
  -> 8. Define the observed-data statistical estimand
  -> 9. Prespecify and run estimators
  -> 10. Run diagnostics and sensitivity analyses
  -> 11. Interpret and report without exceeding the assumptions
```

An effect is not executable merely because SQL can produce treatment and outcome
columns. Steps 1-7 must either pass human review or conclude that the effect is
not identifiable/estimable with the available data.

## Core artifacts

### 1. Causal question

The question records the target population, well-defined treatment strategies,
comparator and outcome. Ambiguous interventions must be refined before moving
forward.

### 2. Target-trial protocol

The protocol records eligibility, treatment strategies, assignment procedure,
time zero, grace period, follow-up, outcome definition, censoring and analysis
strategy. Eligibility, assignment and follow-up must be aligned to prevent
immortal-time and selection biases.

### 3. Causal estimand

The estimand is counterfactual and explicit about population summary (ATE, ATT
or CATE), effect measure, treatment contrast, outcome and horizon. It is distinct
from both a regression coefficient and the estimator used later.

### 4. Causal DAG

The DAG supports exposure, outcome, confounder, mediator, collider, prognostic,
effect-modifier, selection and other nodes. Every node declares whether it is
observed and when it occurs. Every edge has a rationale, evidence identifiers
and review status.

The deterministic layer rejects:

- unknown nodes and duplicate identifiers;
- cycles and self-loops;
- edges that contradict declared temporal order;
- adjustment for exposure, outcome or unobserved variables;
- adjustment for mediators, colliders, selection variables or other descendants
  of treatment.

It checks adjustment sets using d-separation in the exposure backdoor graph and
can enumerate minimum-cardinality valid sets for small reviewable DAGs. Failure
to find a set is a valid scientific result: the effect is not identified by
observed backdoor adjustment under the proposed graph.

### 5. OMOP operationalization

Only after identification is each required DAG variable mapped to:

- an OMOP domain and standard concept set;
- one or more CDM tables and fields;
- an index-relative measurement window;
- a deterministic aggregation rule;
- missingness and provenance rules;
- human mapping status.

OMOP availability must not determine the DAG. An important unobserved confounder
stays in the causal model and becomes an identification limitation; it must not
silently disappear because no convenient column exists.

## Role of LLM, literature retrieval and MCP

- The LLM is a co-pilot that proposes structured artifacts and critiques known
  bias patterns. It never declares its own output identified or approved.
- PubMed and OpenAlex, queried together through `LiteratureOrchestrator`,
  ground clinical relationships and edge rationales. A citation supports an
  assertion; it does not prove the whole DAG is correct.
- MCP/OMCP provides governed metadata, vocabulary lookup and aggregate data
  audits after the scientific specification is established.
- Deterministic code validates schemas, temporal consistency, acyclicity,
  backdoor adjustment, execution gates and artifact provenance.
- Human reviewers approve the question, target trial, DAG/identification and
  OMOP operationalization at separate gates.

## Migration from the current prototype

The existing `HypothesisAgent.candidate_dag` is a prototype limited to
`confounder_to_treatment` and `confounder_to_outcome` edges. It cannot express a
general graph or establish identification. The hard-coded Data Science Agent
also mixes scientific variables with study-specific OMOP mappings.

Migration should proceed incrementally:

1. Keep existing prototype runs readable during the transition.
2. Make `causal.roadmap.CausalStudySpecification` the pre-OMOP contract.
3. Replace the single protocol gate with question, target-trial and
   DAG/identification review gates.
4. Build an OMOP variable registry from approved DAG nodes rather than fixed
   phenotype names.
5. Allow feasibility queries only after the required variables and windows are
   explicit.
6. Add the observed-data estimand and estimator plan only after identification
   and data-support audits pass.

Literature-retrieval and MCP code remain reusable, but they move under the
appropriate Roadmap stage instead of defining the pipeline order.

## Current implementation boundary

The deterministic causal foundation is implemented in `causal/roadmap.py`:

- causal question, target trial and causal estimand contracts;
- general DAG nodes and edges;
- temporal and acyclicity checks;
- backdoor adjustment-set assessment;
- bounded enumeration of minimum adjustment sets;
- explicit assumptions still requiring scientific review.

It is not yet wired into the filesystem workflow or the LLM prompt. Until that
integration is complete, existing workflow commands remain prototype commands
and must not be presented as the final causal pipeline.
