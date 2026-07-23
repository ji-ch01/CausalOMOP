# Demonstration guide (English)

This is the English, LLM/literature-aware companion to
[docs/responsible_demo_guide.md](responsible_demo_guide.md). Same rule
applies: never present a technical or feasibility result as a causal
conclusion. Every example below is copied from a real run already stored
under `artifacts/runs/` in this repository, **except where marked
"illustrative"** — those show the current output *shape* after the
PubMed+OpenAlex change, from before a fresh real run was regenerated against
the current code.

## Elevator pitch (30 seconds, memorize this)

> "CausalOMOP turns a natural-language clinical question into a governed,
> auditable causal-inference workflow over OMOP CDM data. Three agents:
> a Hypothesis Agent — the only place an LLM is allowed to run — drafts the
> causal question and a literature-grounded confounder graph; a Data
> Science Agent maps that draft to real OMOP concepts through typed,
> access-controlled tools, never free SQL; an Execution Agent runs the
> actual causal estimation deterministically, zero LLM involved. Two human
> approval gates sit between these stages, and nothing — not the LLM, not
> any agent — can skip them."

## Narrative arc for the live walkthrough

Don't present this as eight CLI commands — present it as one story with six
beats. Each beat below maps to a numbered section further down this guide.

1. **The problem.** Causal inference from EHR/claims data is easy to get
   wrong — residual confounding, immortal time bias, an ill-defined causal
   question — and usually only an interdisciplinary team catches these in
   time. This design is inspired by Alaa et al. (2024), *"LLMs as Co-Pilots
   for Causal Inference in Medical Studies"*, but adds much stricter
   guardrails than that paper's exploratory demonstrations: every LLM output
   here is a draft, never a decision.
2. **Question → protocol** (§2). A free-text clinical question becomes a
   structured, human-reviewable target trial. `executable: false` always.
3. **Literature grounding, not model memory** (§2–3). Confounders are
   proposed only from abstracts the system actually retrieved (PubMed +
   OpenAlex) — a citation the model wasn't shown gets mechanically rejected.
4. **Identification, cross-validated** (§3). A custom backdoor-adjustment
   algorithm (no external causal library) computes the minimum adjustment
   set — and you can paste the same graph into dagitty.net, the reference
   tool in the field, and get the same answer.
5. **Governed progression** (§4–7). Every step that touches real OMOP data
   or moves toward execution sits behind an explicit human gate; provisional
   phenotypes and unresolved mappings are surfaced, never hidden.
6. **The honest boundary** (§8–9). State plainly what doesn't exist yet: the
   causal estimator itself. This isn't a weakness to hide — it's what makes
   the "governed" claim credible: nothing here overclaims.

## Anticipated questions (English, ready to use)

**"Why not just use an existing library like DoWhy?"**
> "DoWhy is a full causal-inference framework — heavier dependencies
> (scipy, sklearn, statsmodels, sympy), and it would fold the identification
> logic and the future estimator into one adoption decision. We reimplement
> backdoor identification natively in about 100 lines with zero external
> dependency, and get a free cross-validation by exporting to dagitty, the
> field's reference tool, instead."

**"How do you stop the LLM from hallucinating a citation?"**
> "Retrieval happens first, deterministically, against the real PubMed and
> OpenAlex APIs. The second LLM call only sees the abstracts that were
> actually returned. Any citation id in the model's response that isn't in
> that retrieved set is dropped by a plain equality check in Python — not by
> asking the model to be careful. It's covered by an automated test:
> `test_rejects_hallucinated_pmid`."

**"Is this ready for clinical use?"**
> "No, and the repo says so explicitly. What's real: a governed protocol
> draft, literature-grounded confounders, backdoor identification, and
> aggregate feasibility counts against a real OMOP dataset. What's missing:
> a validated Sepsis-3 phenotype, resolved censoring, and — the biggest gap —
> the causal estimator itself. G-computation for CATE first, then TMLE with
> SuperLearner, is the next milestone, not yet written."

**"What happens with the LLM in a real hospital deployment?"**
> "The Hypothesis Agent is the only place an LLM can run, and even there
> it never approves anything itself. For production, `CAUSALOMOP_ENV=prod`
> requires a local, on-premise model and the system refuses to start if a
> cloud API key is still configured — no silent fallback to the cloud. That
> local path is implemented but not yet validated against real on-site
> infrastructure."

**"How do you handle covariates the model isn't confident about?"**
> "Two tiers, kept visually distinct in the DAG export: literature-grounded
> confounders with a PMID citation, and the model's first-pass covariate
> guesses — age, SOFA score, and so on — which never went through retrieval.
> Both are shown to the human reviewer; neither is silently merged into an
> approved adjustment set."

**"What's the single biggest thing left to build?"**
> "The actual estimator. Everything up to here — protocol, DAG,
> identification, feasibility — is real and tested. G-computation for CATE,
> then TMLE with SuperLearner nuisance models, is the next piece, and it
> stays zero-LLM by design: deterministic, reproducible, versioned code."

## 0. Framing (say this before running anything)

> "This is a governed pipeline, not a finished causal-inference result.
> Every LLM output is a draft that a human must approve before it can affect
> data access, and no causal effect has been estimated yet. What you'll see
> is: a question turned into a protocol, a literature-grounded confounder
> list, and a feasibility check — not a treatment-effect number."

## 1. Setup checks

```powershell
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -v
python -m causal.cli config --json
python -m causal.cli preflight --count-persons
```

**Expected output:** all tests pass (81 at the time of writing), the three
BigQuery dataset roles print as configured, and `preflight` reports the 10
required OMOP tables accessible plus a person count (259,770 in the
reference MIMIC-IV OMOP conversion used during development).

## 2. Turn a clinical question into a governed draft (the LLM-facing step)

```powershell
python -m causal.cli workflow start-question "Among adult ICU patients with sepsis, what is the effect of albumin plus crystalloids versus crystalloids alone on 28-day mortality?"
```

This runs **two** LLM calls (both through `openai/gpt-oss-120b` on
OpenRouter in dev) and one deterministic literature retrieval in between —
`LiteratureOrchestrator` queries **both PubMed and OpenAlex** with the same
term, deduplicates by title, and hands the merged set to the second LLM
call. Each raw model response prints to stderr live, tagged by stage
(`target_trial_draft`, then `literature_dag_draft`), so the audience sees
exactly what the model returned, unedited.

### Expected output 1 — `protocol.json`

The normalized target trial. Real example from
`artifacts/runs/albumin_sepsis_icu_mortality-b2b7346f/protocol.json`:

```json
"target_trial": {
  "population": "Adults (≥18 years) admitted to intensive care unit with sepsis (Sepsis-3 definition) receiving initial fluid resuscitation",
  "treatment": "Albumin plus crystalloid fluid resuscitation",
  "comparator": "Crystalloid fluid resuscitation alone",
  "outcome": "28-day all-cause mortality",
  "follow_up_days": 28,
  "primary_estimand": "ATE"
}
```
Every `concept_sets` entry is forced to `"status": "pending_omop_mapping"` —
say out loud that this is not the model being cautious, it's code that makes
it impossible for the LLM to mark its own mapping as done.

### Expected output 2 — `hypothesis.json`

The same content, re-validated by the deterministic `HypothesisAgent`. Point
at `"executable": false` and the four `unresolved_issues` entries
(`concept_set:population:pending_omop_mapping`, ...). Say: *"This is a
safety property, not a bug — an LLM draft can never self-approve."*

### Expected output 3 — literature grounding, inside the same `protocol.json`

*(illustrative — shows the merged PubMed+OpenAlex shape; regenerate a fresh
run before presenting to get real numbers for this exact section)*

```json
"literature_evidence": {
  "search_query": "(\"sepsis\"[Title/Abstract] OR \"ICU\"[Title/Abstract] OR \"adult\"[Title/Abstract]) AND (\"albumin\"[Title/Abstract] OR \"crystalloid\"[Title/Abstract] OR \"fluid resuscitation\"[Title/Abstract]) AND (...)",
  "retrieved_citations": [
    {"source": "pubmed", "id": "42472002"}, {"source": "pubmed", "id": "42469645"},
    {"source": "openalex", "id": "W4392469645"}
  ]
},
"candidate_dag": [
  {
    "confounder": "baseline serum albumin level (hypoalbuminemia)",
    "relation": "confounder_to_treatment",
    "pmid": "42469645",
    "rationale": "The study notes that hypoalbuminemia at ICU admission is a key clinical feature in septic patients, suggesting it influences the decision to add albumin to fluid resuscitation."
  }
]
```
Say: *"Every edge here cites a citation ID that was actually fetched from
PubMed or OpenAlex for this question. If the model cites one it was never
shown, that edge is dropped by a plain Python check — not by asking the
model to behave."* You can show the rejection path live by editing a
citation id in a copy of the JSON and re-running `_parse_dag_response` in a
REPL, or just state it — it's covered by
`tests/test_user_question_hypothesis.py::test_rejects_hallucinated_pmid`.

### Expected output 4 — `llm_outputs.json`

The exact, unedited text both model calls returned, tagged
`target_trial_draft` / `literature_dag_draft`. This is what makes the draft
auditable: nothing between the model and this file is silently rewritten.

## 3. Identification check and DAG visualization

`causal/agents/dag_identification.py` answers a question nothing else in the
pipeline checks: *given only the literature-grounded confounders, is the
effect identified by backdoor adjustment, and what's the minimum set?* This
runs entirely on clinical-language nodes — no OMOP concept is involved yet.

```powershell
python -m causal.cli workflow dag-export <run_id> --highlight-adjustment-set
```

This prints a [dagitty](https://dagitty.net) text export to stdout (paste it
into dagitty.net → *Model > Model code* for an interactive diagram in the
browser — no plotting library or system binary needed) and the
identification summary to stderr.

The exported DAG always includes the exposure → outcome edge itself (tagged
`[proposed]` — it's the hypothesis being tested, not a finding), plus two
tiers of confounders, distinguishable by the `[evidence]` / `[proposed]` tag
on each edge comment:
- **`[evidence]`**: from `candidate_dag`, grounded in a retrieved PubMed or
  OpenAlex abstract, carries that source's citation id (comment still says
  "PMID" for legacy reasons — it may be a PubMed PMID or an OpenAlex work id
  like `W2741809807`; the two never collide since PubMed ids are digit-only).
- **`[proposed]`**: from `first_pass_covariates` (the LLM's first-pass
  covariate list — age, sex, SOFA score, ...), never checked against
  literature, linked to both exposure and outcome by construction since
  "confounder" is exactly what they were proposed as.

**Real output on the `albumin_sepsis_icu_mortality-b2b7346f` run:**
`identified_by_backdoor_adjustment=True confounder_count=15` (5
literature-grounded + 10 first-pass). `--highlight-adjustment-set` marks 14
of the 15 confounders `[adjusted]`. `serum magnesium level` is the sole
exception: it only has a `confounder_to_outcome` edge, no
`confounder_to_treatment` edge, so it isn't on a backdoor path and isn't
structurally required — a genuine, non-trivial result the graph algorithm
computed, not something hand-picked for the demo. Each edge line carries its
rationale (and PMID, when grounded) as a trailing comment, so the pasted
diagram stays traceable back to its source.

## 4. Human gate: protocol approval

```powershell
python -m causal.cli workflow gate <run_id> protocol approved --reviewer "your-name"
python -m causal.cli workflow advance <run_id>
```

**Expected output:** the printed `state.json` shows
`gates.protocol.status = "approved"` and `phase` advances from
`protocol_review` to `concept_mapping`.

## 5. Data Science Agent

```powershell
python -m causal.cli workflow plan <run_id> --with-mcp
```

`--with-mcp` retrieves review-only OMOP concept candidates through OMCP; it
never approves a mapping.

### Expected output — `analysis_plan.json`

Real example (`artifacts/runs/albumin_sepsis_28d_mimic_omop-1590edc5`),
shown here via its `demo_report.md`:

| Phenotype | Role | Mode | Status |
|---|---|---|---|
| sepsis | population | standard | candidate_pending_human_approval |
| albumin | treatment | source_value_provisional | present_but_unmapped_to_standard_concept |
| crystalloids | treatment_and_comparator | source_value_provisional | partially_mapped_requires_product_rules |
| all_cause_death | outcome | standard | standard_table |

`exploratory_execution_allowed` and `final_execution_allowed` are both
`false` until every phenotype above is `approved`.

## 6. Feasibility, cohort SQL, aggregate profile

```powershell
python -m causal.cli workflow gate <run_id> clinical_mapping provisional --reviewer "your-name" --comment "Exploratory feasibility only"
python -m causal.cli workflow advance <run_id>
python -m causal.cli workflow feasibility <run_id> --transport mcp
python -m causal.cli workflow advance <run_id>
python -m causal.cli workflow execution-readiness <run_id>
python -m causal.cli workflow cohort-plan <run_id>
python -m causal.cli workflow cohort-profile <run_id>
python -m causal.cli workflow followup-audit <run_id>
```

### Expected outputs (real numbers from the same run)

- `feasibility.json`: **3,048** eligible persons; **416** with albumin plus
  crystalloids; **2,496** with crystalloids only; **793** deaths within 28
  days across all groups. Every count is prefixed as
  `provisional_source_value_phenotype` — not comparable to a published
  cohort yet.
- `cohort_manifest.json`: `execution_allowed: false`,
  `delivery_to_llm_allowed: false`, `contains_person_level_data: true` — the
  SQL is generated and reviewed, never executed by this gate state.
- `cohort_profile.json` (aggregate-only, via OMCP, zero patient rows
  returned): **2,912** cohort persons, **416** treated, **2,496**
  comparator, **750** outcome events, **319** missing race.
- `followup_audit.json`: **1,454** persons with an observation period
  covering the full 28 days; **733** with incomplete nominal follow-up. This
  is an observability diagnostic, explicitly *not* an approved censoring
  definition.
- `execution_readiness.json`: `causal_estimation_allowed: false`, with an
  itemized `blockers` list (unapproved phenotypes, gates not yet approved,
  covariates not ready) — this is the Execution Agent's policy decision, not
  an estimate.

## 7. Human-readable report

```powershell
python -m causal.cli workflow demo-report <run_id>
Get-Content artifacts\runs\<run_id>\demo_report.md
```

Opens with: *"Architecture and feasibility demonstration only. No causal
effect has been estimated."* Every section it prints is one of the JSON
artifacts above, reformatted — nothing new is computed here.

## 8. What you can present as a result

- A natural-language question converted into a structured, human-reviewable
  protocol, with an auditable LLM trace (`llm_outputs.json`).
- Literature-grounded candidate confounders where every citation is checked
  against PubMed's actual response, not trusted from the model.
- A non-trivial backdoor identification result computed from that literature
  (§3 above).
- Governed, gate-controlled progression from draft to feasibility, with
  provisional phenotypes and unresolved mappings surfaced explicitly at
  every step.
- Real aggregate feasibility counts against a real OMOP dataset (259,770
  persons), retrieved only through the governed OMCP transport.

## 9. What you must not present as a result

- Any cohort size as final.
- Any mortality difference between groups.
- Any ATE, ATT, or CATE.
- Any clinical conclusion about albumin.
- The confounders in §2/§3 as an approved adjustment set — they are a
  literature-grounded *draft*, pending human and clinical review.

These require: Sepsis-3 phenotype implementation, resolved censoring,
non-demographic covariates, validated concept mappings, and — not yet
written — the actual estimator (g-computation, then TMLE/SuperLearner).

## 10. Live-demo risk mitigation

- **Dry-run everything the day before** with your real `.env` — network
  flakiness (OpenRouter, PubMed, BigQuery) during a live talk is the main
  risk, not the code.
- **Keep a pre-generated run as a fallback.** Every command above also works
  read-only against an existing `run_id` (`workflow status`,
  `demo-report`) with no network access needed.
- **`--skip-literature`** on `start-question` if PubMed/OpenAlex is slow or
  unreachable live — falls back to the target-trial draft only, one LLM
  call instead of two.
