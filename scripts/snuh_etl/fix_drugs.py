"""
Fix drug_exposure data quality issues in two passes:

  Pass 1 — Wrong domain: drug_concept_id points to a concept whose
            domain_id != 'Drug' (e.g. Procedure, Observation).
            Action: move concept_id to drug_source_concept_id, set to 0.

  Pass 2 — Non-standard: drug_concept_id points to a non-standard concept
            (standard_concept != 'S', e.g. ICD10CM, LOINC panel, Cost Type).
            Action: zero out and re-remap from drug_source_value via RxNorm
            exact/partial name match.

Both passes are idempotent.
"""

import duckdb
from scripts.snuh_etl.etl_utils import backup_database, connect_write

backup = backup_database("fix_drugs")
print(f"Backup created: {backup}")
con = connect_write()

SEP = "-" * 65


def section(title: str):
    print(f"\n{SEP}\n{title}\n{SEP}")


def ensure_source_concept_col():
    cols = [r[0] for r in con.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_schema='main' AND table_name='drug_exposure'
    """).fetchall()]
    if "drug_source_concept_id" not in cols:
        con.execute(
            "ALTER TABLE main.drug_exposure ADD COLUMN drug_source_concept_id INTEGER DEFAULT 0"
        )


# ── Baseline ──────────────────────────────────────────────────────────────────
section("BASELINE")
total = con.execute("SELECT COUNT(*) FROM main.drug_exposure").fetchone()[0]
print(f"  Total drug records: {total:,}")

con.execute("""
    CREATE TABLE IF NOT EXISTS main.etl_drug_remap_audit (
        script_name VARCHAR,
        drug_source_value VARCHAR,
        old_drug_concept_id INTEGER,
        new_drug_concept_id INTEGER,
        new_concept_name VARCHAR,
        strategy VARCHAR,
        rows_updated BIGINT,
        mapped_at TIMESTAMP DEFAULT current_timestamp
    )
""")
con.execute("BEGIN TRANSACTION")

# ═══════════════════════════════════════════════════════════════════════════════
# PASS 0 — concept_id not found in loaded vocabulary
# ═══════════════════════════════════════════════════════════════════════════════
section("PASS 0 - Concept IDs missing from loaded vocabulary")

missing_concepts = con.execute("""
    SELECT drug_concept_id, COUNT(*) AS n
    FROM main.drug_exposure d
    LEFT JOIN main.concept c ON d.drug_concept_id = c.concept_id
    WHERE d.drug_concept_id != 0
      AND c.concept_id IS NULL
    GROUP BY drug_concept_id
    ORDER BY n DESC
""").fetchall()

if not missing_concepts:
    print("  No non-zero drug_concept_id missing from concept table.")
else:
    total_missing_concept = sum(n for _, n in missing_concepts)
    print(f"  Missing concept IDs : {len(missing_concepts):,}")
    print(f"  Rows affected       : {total_missing_concept:,}")

    ensure_source_concept_col()
    con.execute("""
        INSERT INTO main.etl_drug_remap_audit (
            script_name, drug_source_value, old_drug_concept_id,
            new_drug_concept_id, new_concept_name, strategy, rows_updated
        )
        SELECT
            'fix_drugs.py',
            d.drug_source_value,
            d.drug_concept_id,
            0,
            NULL,
            'missing_concept_zeroed',
            COUNT(*)
        FROM main.drug_exposure d
        LEFT JOIN main.concept c ON d.drug_concept_id = c.concept_id
        WHERE d.drug_concept_id != 0
          AND c.concept_id IS NULL
        GROUP BY d.drug_source_value, d.drug_concept_id
    """)
    con.execute("""
        UPDATE main.drug_exposure
        SET drug_source_concept_id = drug_concept_id,
            drug_concept_id = 0
        WHERE drug_concept_id IN (
            SELECT d.drug_concept_id
            FROM main.drug_exposure d
            LEFT JOIN main.concept c ON d.drug_concept_id = c.concept_id
            WHERE d.drug_concept_id != 0
              AND c.concept_id IS NULL
        )
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# PASS 1 — Wrong domain
# ═══════════════════════════════════════════════════════════════════════════════
section("PASS 1 — Wrong-domain concepts (domain_id != 'Drug')")

wrong_domain = con.execute("""
    SELECT c.domain_id, c.vocabulary_id, COUNT(*) AS n
    FROM main.drug_exposure t
    JOIN main.concept c ON t.drug_concept_id = c.concept_id
    WHERE t.drug_concept_id != 0
      AND c.standard_concept = 'S'
      AND c.domain_id != 'Drug'
    GROUP BY c.domain_id, c.vocabulary_id
    ORDER BY n DESC
""").fetchall()

if not wrong_domain:
    print("  No wrong-domain drug records — already clean.")
else:
    total_wrong = sum(n for *_, n in wrong_domain)
    for domain, vocab, n in wrong_domain:
        print(f"  domain={domain:<20} vocab={vocab:<25} n={n:,}")
    print(f"\n  Total: {total_wrong:,}")

    ensure_source_concept_col()
    con.execute("""
        INSERT INTO main.etl_drug_remap_audit (
            script_name, drug_source_value, old_drug_concept_id,
            new_drug_concept_id, new_concept_name, strategy, rows_updated
        )
        SELECT
            'fix_drugs.py',
            drug_source_value,
            drug_concept_id,
            0,
            NULL,
            'wrong_domain_zeroed',
            COUNT(*)
        FROM main.drug_exposure
        WHERE drug_concept_id IN (
            SELECT concept_id FROM main.concept
            WHERE standard_concept = 'S' AND domain_id != 'Drug'
        )
        AND drug_concept_id != 0
        GROUP BY drug_source_value, drug_concept_id
    """)
    con.execute("""
        UPDATE main.drug_exposure
        SET drug_source_concept_id = drug_concept_id,
            drug_concept_id = 0
        WHERE drug_concept_id IN (
            SELECT concept_id FROM main.concept
            WHERE standard_concept = 'S' AND domain_id != 'Drug'
        )
        AND drug_concept_id != 0
    """)
    print(f"  Fixed {total_wrong:,} records → drug_concept_id set to 0")

# ═══════════════════════════════════════════════════════════════════════════════
# PASS 2 — Non-standard concepts
# ═══════════════════════════════════════════════════════════════════════════════
section("PASS 2 — Non-standard concepts (standard_concept != 'S')")

nonstandard = con.execute("""
    SELECT c.vocabulary_id, c.standard_concept, COUNT(*) AS n
    FROM main.drug_exposure t
    JOIN main.concept c ON t.drug_concept_id = c.concept_id
    WHERE t.drug_concept_id != 0
      AND (c.standard_concept != 'S' OR c.standard_concept IS NULL)
    GROUP BY c.vocabulary_id, c.standard_concept
    ORDER BY n DESC
""").fetchall()

if not nonstandard:
    print("  No non-standard drug concepts — already clean.")
else:
    total_ns = sum(n for *_, n in nonstandard)
    for vocab, std, n in nonstandard:
        print(f"  vocab={vocab:<25} standard={str(std):<6} n={n:,}")
    print(f"\n  Total: {total_ns:,}")

    ensure_source_concept_col()
    con.execute("""
        INSERT INTO main.etl_drug_remap_audit (
            script_name, drug_source_value, old_drug_concept_id,
            new_drug_concept_id, new_concept_name, strategy, rows_updated
        )
        SELECT
            'fix_drugs.py',
            drug_source_value,
            drug_concept_id,
            0,
            NULL,
            'nonstandard_zeroed',
            COUNT(*)
        FROM main.drug_exposure
        WHERE drug_concept_id IN (
            SELECT concept_id FROM main.concept
            WHERE standard_concept != 'S' OR standard_concept IS NULL
        )
        AND drug_concept_id != 0
        GROUP BY drug_source_value, drug_concept_id
    """)
    con.execute("""
        UPDATE main.drug_exposure
        SET drug_source_concept_id = drug_concept_id,
            drug_concept_id = 0
        WHERE drug_concept_id IN (
            SELECT concept_id FROM main.concept
            WHERE standard_concept != 'S' OR standard_concept IS NULL
        )
        AND drug_concept_id != 0
    """)
    print(f"  Zeroed {total_ns:,} non-standard concept_ids")

    # Re-remap from drug_source_value
    print("\n  Re-mapping from drug_source_value...")
    con.execute("DROP TABLE IF EXISTS main._drug_remap")
    con.execute("""
        CREATE TABLE main._drug_remap AS
        SELECT DISTINCT
            d.drug_source_value,
            c.concept_id,
            c.concept_name,
            c.vocabulary_id,
            'exact_name' AS strategy
        FROM (
            SELECT DISTINCT drug_source_value
            FROM main.drug_exposure
            WHERE drug_concept_id = 0
              AND drug_source_value IS NOT NULL AND drug_source_value != ''
        ) d
        JOIN main.concept c
          ON LOWER(TRIM(d.drug_source_value)) = LOWER(TRIM(c.concept_name))
        WHERE c.standard_concept = 'S'
          AND c.invalid_reason IS NULL
          AND c.domain_id = 'Drug'
          AND c.vocabulary_id IN ('RxNorm', 'RxNorm Extension')
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY d.drug_source_value
            ORDER BY
                CASE c.vocabulary_id WHEN 'RxNorm' THEN 1 ELSE 2 END,
                CASE c.concept_class_id
                    WHEN 'Ingredient'    THEN 1
                    WHEN 'Clinical Drug' THEN 2
                    WHEN 'Branded Drug'  THEN 3
                    ELSE 4
                END,
                c.concept_id
        ) = 1
    """)
    n_exact = con.execute("SELECT COUNT(*) FROM main._drug_remap").fetchone()[0]
    print(f"    Strategy A (exact name): {n_exact} source values matched")

    con.execute("""
        INSERT INTO main._drug_remap
        SELECT DISTINCT
            d.drug_source_value,
            c.concept_id,
            c.concept_name,
            c.vocabulary_id,
            'partial_name'
        FROM (
            SELECT DISTINCT drug_source_value
            FROM main.drug_exposure
            WHERE drug_concept_id = 0
              AND drug_source_value IS NOT NULL AND drug_source_value != ''
        ) d
        JOIN main.concept c
          ON LOWER(c.concept_name) LIKE '%' || LOWER(TRIM(d.drug_source_value)) || '%'
        WHERE c.standard_concept = 'S'
          AND c.invalid_reason IS NULL
          AND c.domain_id = 'Drug'
          AND c.vocabulary_id = 'RxNorm'
          AND LENGTH(d.drug_source_value) >= 5
          AND d.drug_source_value NOT IN (SELECT drug_source_value FROM main._drug_remap)
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY d.drug_source_value
            ORDER BY
                CASE c.concept_class_id
                    WHEN 'Ingredient'    THEN 1
                    WHEN 'Clinical Drug' THEN 2
                    ELSE 3
                END,
                LENGTH(c.concept_name),
                c.concept_id
        ) = 1
    """)
    n_partial = con.execute("SELECT COUNT(*) FROM main._drug_remap").fetchone()[0] - n_exact
    print(f"    Strategy B (partial name): {n_partial} additional source values matched")

    con.execute("""
        INSERT INTO main.etl_drug_remap_audit (
            script_name, drug_source_value, old_drug_concept_id,
            new_drug_concept_id, new_concept_name, strategy, rows_updated
        )
        SELECT
            'fix_drugs.py',
            d.drug_source_value,
            d.drug_source_concept_id,
            r.concept_id,
            r.concept_name,
            r.strategy,
            COUNT(*)
        FROM main.drug_exposure d
        JOIN main._drug_remap r ON d.drug_source_value = r.drug_source_value
        WHERE d.drug_concept_id = 0
        GROUP BY
            d.drug_source_value, d.drug_source_concept_id,
            r.concept_id, r.concept_name, r.strategy
    """)

    con.execute("""
        UPDATE main.drug_exposure
        SET drug_concept_id = r.concept_id
        FROM main._drug_remap r
        WHERE main.drug_exposure.drug_source_value = r.drug_source_value
          AND main.drug_exposure.drug_concept_id = 0
    """)
    con.execute("DROP TABLE IF EXISTS main._drug_remap")

# ── Final summary ──────────────────────────────────────────────────────────────
section("PASS 3 - Standard Drug concepts inconsistent with source value")

con.execute("DROP TABLE IF EXISTS main._drug_semantic_remap")
con.execute("""
    CREATE TABLE main._drug_semantic_remap AS
    WITH source_terms AS (
        SELECT DISTINCT
            drug_source_value,
            regexp_replace(
                regexp_replace(
                    lower(trim(drug_source_value)),
                    '\\s*\\([^)]*\\)\\s*$',
                    ''
                ),
                '\\s+[0-9]+\\s*(mg|g|mcg|ml|iu|unit|units)(/ml)?(\\s+.*)?$',
                ''
            ) AS normalized_source_value
        FROM main.drug_exposure
        WHERE drug_source_value IS NOT NULL
          AND drug_source_value != ''
    )
    SELECT
        s.drug_source_value,
        s.normalized_source_value,
        c.concept_id,
        c.concept_name,
        c.vocabulary_id,
        'normalized_exact_name' AS strategy
    FROM source_terms s
    JOIN main.concept c
      ON lower(c.concept_name) = s.normalized_source_value
    WHERE c.standard_concept = 'S'
      AND c.invalid_reason IS NULL
      AND c.domain_id = 'Drug'
      AND c.vocabulary_id IN ('RxNorm', 'RxNorm Extension')
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY s.drug_source_value
        ORDER BY
            CASE c.vocabulary_id WHEN 'RxNorm' THEN 1 ELSE 2 END,
            CASE c.concept_class_id
                WHEN 'Ingredient'    THEN 1
                WHEN 'Clinical Drug' THEN 2
                WHEN 'Branded Drug'  THEN 3
                ELSE 4
            END,
            c.concept_id
    ) = 1
""")

semantic_sources = con.execute("""
    SELECT COUNT(*)
    FROM main._drug_semantic_remap r
    WHERE EXISTS (
        SELECT 1
        FROM main.drug_exposure d
        WHERE d.drug_source_value = r.drug_source_value
          AND d.drug_concept_id != r.concept_id
    )
""").fetchone()[0]

if semantic_sources == 0:
    print("  No semantically inconsistent exact source mappings found.")
else:
    semantic_rows = con.execute("""
        SELECT COUNT(*)
        FROM main.drug_exposure d
        JOIN main._drug_semantic_remap r
          ON d.drug_source_value = r.drug_source_value
        WHERE d.drug_concept_id != r.concept_id
    """).fetchone()[0]
    print(f"  Source values to correct : {semantic_sources:,}")
    print(f"  Rows to update           : {semantic_rows:,}")

    con.execute("""
        INSERT INTO main.etl_drug_remap_audit (
            script_name, drug_source_value, old_drug_concept_id,
            new_drug_concept_id, new_concept_name, strategy, rows_updated
        )
        SELECT
            'fix_drugs.py',
            d.drug_source_value,
            d.drug_concept_id,
            r.concept_id,
            r.concept_name,
            r.strategy,
            COUNT(*)
        FROM main.drug_exposure d
        JOIN main._drug_semantic_remap r
          ON d.drug_source_value = r.drug_source_value
        WHERE d.drug_concept_id != r.concept_id
        GROUP BY
            d.drug_source_value, d.drug_concept_id,
            r.concept_id, r.concept_name, r.strategy
    """)

    con.execute("""
        UPDATE main.drug_exposure
        SET drug_source_concept_id = CASE
                WHEN drug_concept_id != 0 THEN drug_concept_id
                ELSE drug_source_concept_id
            END,
            drug_concept_id = r.concept_id
        FROM main._drug_semantic_remap r
        WHERE main.drug_exposure.drug_source_value = r.drug_source_value
          AND main.drug_exposure.drug_concept_id != r.concept_id
    """)

con.execute("DROP TABLE IF EXISTS main._drug_semantic_remap")

section("SUMMARY")
unmapped_final = con.execute(
    "SELECT COUNT(*) FROM main.drug_exposure WHERE drug_concept_id = 0"
).fetchone()[0]
standard_final = con.execute("""
    SELECT COUNT(*) FROM main.drug_exposure t
    JOIN main.concept c ON t.drug_concept_id = c.concept_id
    WHERE t.drug_concept_id != 0 AND c.standard_concept = 'S' AND c.domain_id = 'Drug'
""").fetchone()[0]
still_nonstandard = con.execute("""
    SELECT COUNT(*) FROM main.drug_exposure t
    JOIN main.concept c ON t.drug_concept_id = c.concept_id
    WHERE t.drug_concept_id != 0 AND (c.standard_concept != 'S' OR c.standard_concept IS NULL)
""").fetchone()[0]

print(f"  Total                : {total:>10,}")
print(f"  Standard Drug (S)    : {standard_final:>10,}  ({standard_final/total*100:.1f}%)")
print(f"  Unmapped (cid=0)     : {unmapped_final:>10,}  ({unmapped_final/total*100:.1f}%)")
print(f"  Non-standard remain  : {still_nonstandard:>10,}")

con.execute("COMMIT")
con.close()
print("\nDone.")
