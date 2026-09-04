# Vocabulary Mapping Coverage: A Source-Data Limitation

Dated technical note, intended as source material for the manuscript's
limitations section. Concerns the source-to-standard vocabulary mapping
completeness of the underlying OMOP CDM conversion, not any property of
CausalOMOP's own query or estimation logic.

## Summary

Two independent OMOP CDM clinical-event tables in the current MIMIC-IV
conversion were audited for standard-vocabulary mapping completeness. A
substantial share of rows in both tables carry `concept_id = 0`
("no matching concept" in the CDM specification) rather than a standard
concept -- meaning the corresponding clinical event exists in the source
data but is invisible to any query that selects on standard `concept_id`s,
as every extraction in this pipeline does. One table's overall unmapped
rate improved by an order of magnitude after a vocabulary configuration
update, but coverage remained highly uneven across specific drug
products even after that update, and the second table's gap has not been
addressed at all. This is reported as a property of the data conversion
this project draws on, evaluated at two points in time, not resolved by
this pipeline itself. A third, distinct limitation -- the source
`inputevents` table is not staged at all, so intensive-care infusions are
absent or order-only regardless of vocabulary coverage -- is described
under "Infusion administrations" below.

## Background: what "unmapped" means in OMOP CDM

The OMOP Common Data Model represents every clinical fact with both a
`_source_value` (the free-text or local-code string as recorded by the
source system) and a `_concept_id` (a standard, cross-vocabulary concept
the ETL is expected to resolve it to). `concept_id = 0` is the CDM's
explicit convention for "no matching concept was found" -- the row is
present, but not queryable through the standard vocabulary
(OHDSI/CommonDataModel, https://github.com/OHDSI/CommonDataModel). Any
pipeline, including this one, that builds cohorts by expanding an
approved concept set (root concepts plus `concept_ancestor` descendants)
will silently miss every row stuck at `concept_id = 0`, however clinically
common the underlying event actually is.

## Method used to quantify coverage

Two complementary diagnostics were used, both aggregate-only (no
patient-level data leaves the database):

1. **Domain-wide unmapped rate**: `COUNTIF(concept_id = 0) / COUNT(*)`
   over an entire clinical table -- a coarse hygiene signal for the table
   as a whole.
2. **Concept-level coverage**: for one audited concept (root concept plus
   its standard descendants), the count of rows actually carrying one of
   those standard `concept_id`s, compared against the count of rows with
   `concept_id = 0` whose `_source_value` free text still names that same
   concept (a `LIKE` match on the concept's own vocabulary name). A large
   gap between these two counts is evidence of a systematic mapping gap
   for that concept specifically, rather than the concept being genuinely
   rare in the population -- a genuinely rare but well-mapped concept
   would show few rows on both sides, not many unmapped rows with none
   mapped. This text-match check is the same class of diagnostic as the
   orphan-concept and index-event-breakdown checks in OHDSI's
   CohortDiagnostics methodology (Rao GA, Shoaibi A, Makadia R, et al.
   "CohortDiagnostics: Phenotype evaluation across a network of
   observational data sources using population-level characterization."
   *PLOS ONE*. 2025;20(1):e0310634. doi:10.1371/journal.pone.0310634),
   adapted here to a mapping-coverage question rather than a phenotype
   orphan-concept question.

## Findings

### Drug-domain table: large improvement, but uneven across products

- At an earlier audit point, 86.9% of all rows in the drug-administration
  table carried `concept_id = 0`.
- After a subsequent vocabulary configuration update, the same table-wide
  rate fell to 3.2% -- roughly a 27-fold reduction, with the total row
  count essentially unchanged, consistent with a re-mapping of existing
  rows rather than a data reload.
- Concept-level coverage for one commonly prescribed oral-formulary drug
  ingredient, audited in detail as a representative case, moved from
  effectively zero (a handful of mapped rows against tens of thousands of
  unmapped rows whose source text named that same ingredient) to
  nearly complete (order of 10^5 mapped rows against a low double-digit
  count of remaining unmapped, name-matched rows).
- However, at least one other clinically relevant drug product's
  concept-level coverage remained at zero after the same update -- every
  row naming that product in its source text still carried
  `concept_id = 0`. The improvement in the table-wide rate is therefore
  real but **heterogeneous**: it does not imply that any given drug
  concept a study happens to need is now adequately mapped, only that
  the table as a whole is markedly less affected than before.

### Measurement-domain table: a separate, still-unaddressed gap

- The measurement table's table-wide unmapped rate is 5.85% -- lower than
  the drug table's rate ever was, but applied to a much larger table, so
  the absolute row count affected is large. No earlier audit of this
  table exists for comparison, so whether this rate has changed over time
  is unknown.
- Unlike the drug table, the most frequent unmapped rows are not
  identifiable by their source text: their `_source_value` is a bare
  numeric source-system item identifier with no accompanying label,
  several of them individually accounting for over a million rows. This
  means the affected measurements cannot be characterized from the OMOP
  tables alone -- confirming which specific lab tests or vital signs they
  represent would require the source system's own item dictionary, which
  is outside the converted OMOP dataset. This gap has not been touched by
  the vocabulary update described above and remains fully open.

### Infusion administrations: a table-scope gap, not a mapping gap

Separate from `concept_id = 0` coverage, the OHDSI/MIMIC conversion this
project reads does not stage the source `inputevents` table at all. Drug
exposures are derived from prescription **orders** (`drug_type_concept_id =
32838`): an order carries an order time but no dose, no infusion rate, and no
administration/infusion-start timestamp. The consequence is that clinical
events given predominantly or exclusively as intravenous infusions are
either **absent entirely** or present only as an order with no quantitative
detail -- among them resuscitation crystalloids and colloids (albumin),
blood products, titrated vasopressors, sedation infusions, and intravenous
insulin. Standard vocabulary concepts for these substances exist and are
loaded; the gap is that the source rows describing the infusions are never
converted, so no concept set -- however well mapped -- can recover them.

This is a property of the source conversion's table scope, verified by
direct audit of the converted dataset (zero rows via `concept_ancestor`
expansion for infusion-only ingredients; `d_items_to_concept.csv` carries
the infusion itemids only as unmapped stubs). It matters for any
intensive-care causal question whose exposure, time zero, or a key covariate
is an infusion: such a question cannot be operationalised on the current ETL
and needs an `inputevents` extension upstream first. `causal/queries/`
feasibility checks and the mapping audit surface a zero-patient or
order-only exposure arm rather than proceeding silently.

## Implication for causal validity

A concept set built only from standard `concept_id`s can undercount an
exposure, outcome, or covariate whose source rows are unmapped, without
raising any error -- the query simply returns fewer eligible or exposed
persons than actually exist in the source data. This is a silent
threat to internal validity distinct from confounding or model
misspecification: a cohort can look well-defined and well-powered while
still materially undercounting the population of interest, purely as an
artifact of ETL vocabulary coverage rather than of the causal
identification strategy.

## Mitigation already implemented

Because concept-level coverage varies by concept and can change between
audits, CausalOMOP checks it automatically, per concept set, immediately
before any cohort-defining query executes -- rather than relying on a
one-off manual audit like the one this note reports. The check compares
mapped-row count against unmapped-but-name-matched-row count (the same
method described above) for every role (population, exposure, comparator,
outcome) whose concept set touches a concept-coded domain, and blocks
cohort construction with an explicit error when coverage falls below a
fixed floor, once enough rows exist on either side to be informative.
This turns what would otherwise be a silently underpowered or
zero-person cohort into an explicit, actionable failure at query time.

## Residual limitation to state in the manuscript

The mitigation above detects the problem; it does not fix the source
data. Any study drawing on a concept for which coverage is genuinely low
in this dataset build cannot proceed until either the concept is
re-mapped upstream (in the MIMIC-IV-to-OMOP conversion itself) or a
different, better-mapped proxy concept is substituted -- both of which
are data-engineering decisions outside this project's causal-inference
scope. The manuscript should state plainly that **vocabulary mapping
completeness in the source OMOP conversion is a precondition for valid
cohort ascertainment that this pipeline can verify and enforce, but
cannot itself guarantee** -- and that mapping completeness should be
re-audited whenever the underlying vocabulary or ETL configuration
changes, not assumed to be stable over the life of a study.

## References

- OHDSI. OMOP Common Data Model specification.
  https://github.com/OHDSI/CommonDataModel
- OHDSI/MIMIC. MIMIC-IV to OMOP CDM 5.3.1 extract-transform-load project.
  https://github.com/OHDSI/MIMIC
- Rao GA, Shoaibi A, Makadia R, et al. CohortDiagnostics: Phenotype
  evaluation across a network of observational data sources using
  population-level characterization. *PLOS ONE*. 2025;20(1):e0310634.
  doi:10.1371/journal.pone.0310634
