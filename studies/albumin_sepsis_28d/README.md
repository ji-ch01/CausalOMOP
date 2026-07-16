# Albumin for sepsis: OMOP replication

This study is the first end-to-end validation case for CausalOMOP. It translates
the published MIMIC-IV v2.2 analysis (DOI `10.1371/journal.pdig.0000721`) into
an OMOP CDM workflow.

The protocol is intentionally marked as a draft. Candidate concepts found by
name are evidence for review, not executable concept sets. Approve seed concepts
and descendant rules before cohort SQL is generated.

## Concept audit

Run from the repository root with the virtual environment active:

```powershell
python -m causal.cli concepts "sepsis" --limit 20
python -m causal.cli concepts "septic shock" --limit 20
python -m causal.cli concepts "albumin" --limit 30
python -m causal.cli concepts "sodium chloride" --limit 30
python -m causal.cli concepts "lactated ringer" --limit 30
python -m causal.cli concept-usage condition "sepsis" --limit 30
python -m causal.cli concept-usage drug "albumin" --limit 30
python -m causal.cli concept-usage drug "sodium chloride" --limit 30
python -m causal.cli source-usage drug "albumin" --limit 30
python -m causal.cli source-usage drug "ringer" --limit 30
```

The next gate is human approval of concept sets. Only after that gate should the
pipeline count the eligible population and treatment arms.
