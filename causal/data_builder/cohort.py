"""
Causal cohort builder — Phenotyping Agent.

Extracts one row per patient with:
    person_id | A (treatment) | Y (outcome) | index_date | follow_up_days | censored

Supports two control strategies:
  - Untreated controls: eligible patients who never received the treatment drug.
      index_date = first diagnosis date of the population condition.
  - Active comparator: eligible patients who received a different comparator drug.
      index_date = first exposure to whichever drug (treatment or comparator) came first.
      Ties (same day) are excluded.

Supports two outcome types:
  - "mortality": death recorded within [index_date, index_date + follow_up_days].
  - "condition": new occurrence of outcome_condition within the same window.

All downstream covariate extractors use index_date as the pre/post boundary.
"""

import duckdb
import pandas as pd
from dataclasses import dataclass
from typing import Optional


DB_PATH = r"data/synthetic_snuh.duckdb"


@dataclass
class CohortDefinition:
    """Parameters describing a single causal question."""
    name: str
    treatment_drug: str         # partial match on concept_name, case-insensitive
    population_condition: str   # partial match on concept_name, case-insensitive
    follow_up_days: int = 28
    comparator_drug: Optional[str] = None   # active comparator; None = untreated controls
    outcome_type: str = "mortality"          # "mortality" | "condition"
    outcome_condition: Optional[str] = None  # required when outcome_type="condition"


# ── Pre-built causal questions ────────────────────────────────────────────────

METFORMIN_T2DM_MORTALITY = CohortDefinition(
    name="metformin_t2dm_28day_mortality",
    treatment_drug="metformin",
    population_condition="Type 2 diabetes mellitus",
    follow_up_days=28,
)

ATORVASTATIN_HL_MORTALITY = CohortDefinition(
    name="atorvastatin_hyperlipidemia_28day_mortality",
    treatment_drug="atorvastatin",
    population_condition="Hyperlipidemia",
    follow_up_days=28,
)

AMLODIPINE_HTN_MORTALITY = CohortDefinition(
    name="amlodipine_hypertension_28day_mortality",
    treatment_drug="amlodipine",
    population_condition="Essential hypertension",
    follow_up_days=28,
)

# ── 4 candidate causal questions ──────────────────────────────────────────────

# Q1 — Atorvastatin vs Rosuvastatin: does choice of statin affect 1-year mortality?
ATORVASTATIN_VS_ROSUVASTATIN_HL = CohortDefinition(
    name="atorvastatin_vs_rosuvastatin_hyperlipidemia_365day_mortality",
    treatment_drug="atorvastatin",
    population_condition="Hyperlipidemia",
    follow_up_days=365,
    comparator_drug="rosuvastatin",
)

# Q2 — Celecoxib vs Meloxicam: does COX-2 selectivity reduce GI ulcer risk?
CELECOXIB_VS_MELOXICAM_OA_GI = CohortDefinition(
    name="celecoxib_vs_meloxicam_osteoarthritis_90day_gi_ulcer",
    treatment_drug="celecoxib",
    population_condition="Osteoarthritis",
    follow_up_days=90,
    comparator_drug="meloxicam",
    outcome_type="condition",
    outcome_condition="Gastrointestinal ulcer",
)

# Q3 — Peramivir vs untreated: does antiviral reduce 14-day mortality in influenza?
PERAMIVIR_INFLUENZA_MORTALITY = CohortDefinition(
    name="peramivir_influenza_14day_mortality",
    treatment_drug="peramivir",
    population_condition="Influenza",
    follow_up_days=14,
)

# Q4 — Dapagliflozin vs Sitagliptin: does SGLT2i reduce 1-year mortality vs DPP4i in T2DM?
DAPAGLIFLOZIN_VS_SITAGLIPTIN_T2DM = CohortDefinition(
    name="dapagliflozin_vs_sitagliptin_t2dm_365day_mortality",
    treatment_drug="dapagliflozin",
    population_condition="Type 2 diabetes mellitus",
    follow_up_days=365,
    comparator_drug="sitagliptin",
)


# ── Core function ─────────────────────────────────────────────────────────────

def build_cohort(
    con: duckdb.DuckDBPyConnection,
    definition: CohortDefinition,
) -> pd.DataFrame:
    """
    Build a causal cohort table from OMOP CDM.

    Returns:
        DataFrame with columns: person_id, A, Y, index_date, follow_up_days, censored
    """
    drug    = definition.treatment_drug.lower()
    cond    = definition.population_condition.lower()
    fw      = definition.follow_up_days

    ctes = []

    # ── Eligible population ───────────────────────────────────────────────────
    ctes.append(f"""eligible_patients AS (
        SELECT DISTINCT co.person_id
        FROM main.condition_occurrence co
        JOIN main.concept c ON co.condition_concept_id = c.concept_id
        WHERE LOWER(c.concept_name) LIKE '%{cond}%'
          AND c.standard_concept = 'S'
    )""")

    # ── First treatment exposure ───────────────────────────────────────────────
    ctes.append(f"""first_treatment AS (
        SELECT de.person_id, MIN(de.drug_exposure_start_date) AS index_date
        FROM main.drug_exposure de
        JOIN main.concept c ON de.drug_concept_id = c.concept_id
        WHERE LOWER(c.concept_name) LIKE '%{drug}%'
          AND c.standard_concept = 'S'
          AND de.person_id IN (SELECT person_id FROM eligible_patients)
        GROUP BY de.person_id
    )""")

    # ── Cohort base: active comparator OR untreated controls ──────────────────
    if definition.comparator_drug:
        comp = definition.comparator_drug.lower()
        ctes.append(f"""first_comparator AS (
        SELECT de.person_id, MIN(de.drug_exposure_start_date) AS comp_date
        FROM main.drug_exposure de
        JOIN main.concept c ON de.drug_concept_id = c.concept_id
        WHERE LOWER(c.concept_name) LIKE '%{comp}%'
          AND c.standard_concept = 'S'
          AND de.person_id IN (SELECT person_id FROM eligible_patients)
        GROUP BY de.person_id
    )""")
        ctes.append("""cohort_base AS (
        SELECT
            COALESCE(ft.person_id, fc.person_id) AS person_id,
            CASE
                WHEN ft.person_id IS NOT NULL AND fc.person_id IS NULL THEN 1
                WHEN ft.person_id IS NULL AND fc.person_id IS NOT NULL THEN 0
                WHEN ft.index_date < fc.comp_date THEN 1
                WHEN fc.comp_date  < ft.index_date THEN 0
                -- same day: NULL → filtered out in final SELECT
            END AS A,
            CASE
                WHEN ft.person_id IS NOT NULL AND fc.person_id IS NULL THEN ft.index_date
                WHEN ft.person_id IS NULL AND fc.person_id IS NOT NULL THEN fc.comp_date
                WHEN ft.index_date < fc.comp_date THEN ft.index_date
                WHEN fc.comp_date  < ft.index_date THEN fc.comp_date
            END AS index_date
        FROM first_treatment ft
        FULL OUTER JOIN first_comparator fc ON ft.person_id = fc.person_id
    )""")
    else:
        ctes.append(f"""first_diagnosis AS (
        SELECT co.person_id, MIN(co.condition_start_date) AS index_date
        FROM main.condition_occurrence co
        JOIN main.concept c ON co.condition_concept_id = c.concept_id
        WHERE LOWER(c.concept_name) LIKE '%{cond}%'
          AND c.standard_concept = 'S'
        GROUP BY co.person_id
    )""")
        ctes.append("""cohort_base AS (
        SELECT
            ep.person_id,
            COALESCE(ft.index_date, fd.index_date) AS index_date,
            CASE WHEN ft.person_id IS NOT NULL THEN 1 ELSE 0 END AS A
        FROM eligible_patients ep
        LEFT JOIN first_treatment ft ON ep.person_id = ft.person_id
        LEFT JOIN first_diagnosis  fd ON ep.person_id = fd.person_id
        WHERE COALESCE(ft.index_date, fd.index_date) IS NOT NULL
    )""")

    # ── Outcome ───────────────────────────────────────────────────────────────
    if definition.outcome_type == "condition":
        out_cond = definition.outcome_condition.lower()
        # New occurrences of outcome_condition strictly after index_date
        ctes.append(f"""outcome_flags AS (
        SELECT DISTINCT co.person_id
        FROM main.condition_occurrence co
        JOIN main.concept c   ON co.condition_concept_id = c.concept_id
        JOIN cohort_base   cb ON co.person_id = cb.person_id
        WHERE LOWER(c.concept_name) LIKE '%{out_cond}%'
          AND c.standard_concept = 'S'
          AND co.condition_start_date >  cb.index_date
          AND co.condition_start_date <= cb.index_date + INTERVAL '{fw} days'
    )""")
        ctes.append(f"""cohort_with_outcome AS (
        SELECT
            cb.person_id,
            cb.A,
            cb.index_date,
            CASE WHEN of.person_id IS NOT NULL THEN 1 ELSE 0 END AS Y,
            {fw} AS follow_up_days,
            CASE WHEN of.person_id IS NOT NULL THEN 0 ELSE 1 END AS censored
        FROM cohort_base cb
        LEFT JOIN outcome_flags of ON cb.person_id = of.person_id
    )""")
    else:
        ctes.append(f"""cohort_with_outcome AS (
        SELECT
            cb.person_id,
            cb.A,
            cb.index_date,
            CASE
                WHEN d.death_date IS NOT NULL
                 AND d.death_date >= cb.index_date
                 AND d.death_date <= cb.index_date + INTERVAL '{fw} days'
                THEN 1 ELSE 0
            END AS Y,
            CASE
                WHEN d.death_date IS NOT NULL
                 AND d.death_date >= cb.index_date
                 AND d.death_date <= cb.index_date + INTERVAL '{fw} days'
                THEN DATEDIFF('day', cb.index_date, d.death_date)
                ELSE {fw}
            END AS follow_up_days,
            CASE
                WHEN d.death_date IS NOT NULL
                 AND d.death_date >= cb.index_date
                 AND d.death_date <= cb.index_date + INTERVAL '{fw} days'
                THEN 0 ELSE 1
            END AS censored
        FROM cohort_base cb
        LEFT JOIN main.death d ON cb.person_id = d.person_id
    )""")

    query = (
        "WITH " + ",\n\n".join(ctes) +
        "\nSELECT person_id, A, Y, index_date, follow_up_days, censored"
        "\nFROM cohort_with_outcome"
        "\nWHERE A IS NOT NULL"
        "\nORDER BY person_id"
    )

    df = con.execute(query).df()
    _print_summary(df, definition)
    return df


def _print_summary(df: pd.DataFrame, definition: CohortDefinition) -> None:
    n                = len(df)
    n_treated        = int(df["A"].sum())
    n_control        = n - n_treated
    n_events         = int(df["Y"].sum())
    n_events_treated = int(df.loc[df["A"] == 1, "Y"].sum())
    n_events_control = int(df.loc[df["A"] == 0, "Y"].sum())

    ctrl_label = (
        f"Control  (A=0, {definition.comparator_drug})"
        if definition.comparator_drug
        else "Control  (A=0, untreated)"
    )
    outcome_label = (
        f"Outcome ({definition.outcome_condition})"
        if definition.outcome_type == "condition"
        else "Outcome (death)"
    )

    print(f"\n{'='*60}")
    print(f"Cohort: {definition.name}")
    print(f"{'='*60}")
    print(f"  Eligible patients    : {n:,}")
    print(f"  Treated  (A=1)       : {n_treated:,}  ({n_treated/n*100:.1f}%)")
    print(f"  {ctrl_label:<30}: {n_control:,}  ({n_control/n*100:.1f}%)")
    print(f"  {outcome_label} (Y=1)  : {n_events:,}  ({n_events/n*100:.2f}%)")
    print(f"    — in treated group : {n_events_treated}")
    print(f"    — in control group : {n_events_control}")
    print(f"  Follow-up window     : {definition.follow_up_days} days")
    print(f"{'='*60}\n")


def scan_cohorts(
    con: duckdb.DuckDBPyConnection,
    definitions: list,
) -> pd.DataFrame:
    """Run all cohorts and return a feasibility summary table."""
    rows = []
    for defn in definitions:
        df = build_cohort(con, defn)
        n                = len(df)
        n_treated        = int(df["A"].sum())
        n_control        = n - n_treated
        n_events         = int(df["Y"].sum())
        n_events_control = int(df.loc[df["A"] == 0, "Y"].sum())
        treat_pct        = n_treated / n * 100 if n > 0 else 0

        positivity_ok  = 10 <= treat_pct <= 90
        events_ok      = n_events >= 20
        ctrl_events_ok = n_events_control >= 5
        feasible       = positivity_ok and events_ok and ctrl_events_ok

        rows.append({
            "cohort"          : defn.name,
            "n"               : n,
            "treated_%"       : f"{treat_pct:.1f}%",
            "n_control"       : n_control,
            "n_events"        : n_events,
            "events_treated"  : n_events - n_events_control,
            "events_control"  : n_events_control,
            "positivity"      : "ok" if positivity_ok  else "FAIL",
            "events>=20"      : "ok" if events_ok      else "FAIL",
            "ctrl_events>=5"  : "ok" if ctrl_events_ok else "FAIL",
            "feasible"        : "YES" if feasible else "no",
        })

    return pd.DataFrame(rows)


# ── Scan all cohorts ──────────────────────────────────────────────────────────

ALL_COHORTS = [
    METFORMIN_T2DM_MORTALITY,
    ATORVASTATIN_HL_MORTALITY,
    AMLODIPINE_HTN_MORTALITY,
    ATORVASTATIN_VS_ROSUVASTATIN_HL,
    CELECOXIB_VS_MELOXICAM_OA_GI,
    PERAMIVIR_INFLUENZA_MORTALITY,
    DAPAGLIFLOZIN_VS_SITAGLIPTIN_T2DM,
]

if __name__ == "__main__":
    con = duckdb.connect(DB_PATH, read_only=True)

    summary = scan_cohorts(con, ALL_COHORTS)

    print("\n" + "=" * 60)
    print("FEASIBILITY SCAN — ALL COHORTS")
    print("=" * 60)
    pd.set_option("display.max_colwidth", 55)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    print("=" * 60)

    con.close()
