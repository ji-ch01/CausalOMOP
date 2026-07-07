"""
Build an auditable extension of concept_ancestor for drug concepts missing from
the Athena hierarchy.

This script deliberately does NOT modify main.concept_ancestor. Athena's table
is treated as source vocabulary truth. Local additions are written to:

  - main.concept_ancestor_ext
  - main.concept_ancestor_augmented  (view = Athena + local extension)

Only true hierarchy-like relationships are used. Mapping relationships such as
'Maps to' are not ancestor relationships and should be handled during concept
normalization, not inserted into concept_ancestor.
"""

from scripts.snuh_etl.etl_utils import backup_database, connect_write


backup = backup_database("extend_concept_ancestor")
print(f"Backup created: {backup}")
con = connect_write()

SEP = "-" * 65


def section(title: str):
    print(f"\n{SEP}\n{title}\n{SEP}")


section("0. BASELINE - concept_ancestor coverage for drug_exposure")
distinct_drug_ids = con.execute("""
    SELECT COUNT(DISTINCT drug_concept_id)
    FROM main.drug_exposure
    WHERE drug_concept_id != 0
""").fetchone()[0]

in_ancestor = con.execute("""
    SELECT COUNT(DISTINCT d.drug_concept_id)
    FROM main.drug_exposure d
    JOIN main.concept_ancestor ca ON d.drug_concept_id = ca.descendant_concept_id
    WHERE d.drug_concept_id != 0
      AND ca.min_levels_of_separation > 0
""").fetchone()[0]

missing = distinct_drug_ids - in_ancestor
pct = in_ancestor / distinct_drug_ids * 100 if distinct_drug_ids > 0 else 0
print(f"  Distinct drug concept_ids in use : {distinct_drug_ids:>8,}")
print(f"  Covered by true ancestors        : {in_ancestor:>8,}  ({pct:.1f}%)")
print(f"  Missing true ancestor coverage   : {missing:>8,}")

section("1. BUILDING LOCAL HIERARCHY EXTENSION")

hierarchy_rels = "'Is a', 'RxNorm is a'"

con.execute("DROP TABLE IF EXISTS main.concept_ancestor_ext")
con.execute(f"""
    CREATE TABLE main.concept_ancestor_ext AS
    SELECT DISTINCT
        cr.concept_id_2 AS ancestor_concept_id,
        cr.concept_id_1 AS descendant_concept_id,
        1 AS min_levels_of_separation,
        1 AS max_levels_of_separation,
        cr.relationship_id,
        'extend_concept_ancestor.py' AS generated_by,
        current_timestamp AS generated_at
    FROM (
        SELECT DISTINCT drug_concept_id AS concept_id
        FROM main.drug_exposure
        WHERE drug_concept_id != 0
          AND drug_concept_id NOT IN (
              SELECT DISTINCT descendant_concept_id
              FROM main.concept_ancestor
              WHERE min_levels_of_separation > 0
          )
    ) missing_drugs
    JOIN main.concept_relationship cr
      ON cr.concept_id_1 = missing_drugs.concept_id
    JOIN main.concept ancestor
      ON ancestor.concept_id = cr.concept_id_2
    WHERE cr.relationship_id IN ({hierarchy_rels})
      AND cr.invalid_reason IS NULL
      AND ancestor.standard_concept = 'S'
      AND ancestor.invalid_reason IS NULL
      AND ancestor.domain_id = 'Drug'
      AND cr.concept_id_2 IN (
          SELECT DISTINCT ancestor_concept_id
          FROM main.concept_ancestor
          WHERE min_levels_of_separation > 0
      )
""")

ext_rows = con.execute("SELECT COUNT(*) FROM main.concept_ancestor_ext").fetchone()[0]
ext_concepts = con.execute("""
    SELECT COUNT(DISTINCT descendant_concept_id)
    FROM main.concept_ancestor_ext
""").fetchone()[0]
print(f"  Extension rows                 : {ext_rows:>8,}")
print(f"  Missing concepts newly covered : {ext_concepts:>8,} / {missing:,}")
print("  Self-ancestor rows are intentionally excluded from coverage scoring.")

section("2. CREATING AUGMENTED VIEW")
con.execute("""
    CREATE OR REPLACE VIEW main.concept_ancestor_augmented AS
    SELECT
        ancestor_concept_id,
        descendant_concept_id,
        min_levels_of_separation,
        max_levels_of_separation,
        'athena' AS source
    FROM main.concept_ancestor
    UNION ALL
    SELECT
        ext.ancestor_concept_id,
        ext.descendant_concept_id,
        ext.min_levels_of_separation,
        ext.max_levels_of_separation,
        ext.generated_by AS source
    FROM main.concept_ancestor_ext ext
    WHERE NOT EXISTS (
        SELECT 1
        FROM main.concept_ancestor ca
        WHERE ca.ancestor_concept_id = ext.ancestor_concept_id
          AND ca.descendant_concept_id = ext.descendant_concept_id
    )
""")
print("  Created main.concept_ancestor_ext")
print("  Created main.concept_ancestor_augmented")

section("3. UPDATED COVERAGE")
in_augmented = con.execute("""
    SELECT COUNT(DISTINCT d.drug_concept_id)
    FROM main.drug_exposure d
    JOIN main.concept_ancestor_augmented ca
      ON d.drug_concept_id = ca.descendant_concept_id
    WHERE d.drug_concept_id != 0
      AND ca.min_levels_of_separation > 0
""").fetchone()[0]

pct_after = in_augmented / distinct_drug_ids * 100 if distinct_drug_ids > 0 else 0
still_missing = distinct_drug_ids - in_augmented
print(f"  Coverage before : {in_ancestor:>8,} / {distinct_drug_ids:,}  ({pct:.1f}%)")
print(f"  Coverage after  : {in_augmented:>8,} / {distinct_drug_ids:,}  ({pct_after:.1f}%)")
print(f"  Still missing   : {still_missing:>8,}")

if still_missing > 0:
    leftover = con.execute("""
        SELECT c.vocabulary_id, COUNT(DISTINCT d.drug_concept_id) AS n
        FROM main.drug_exposure d
        JOIN main.concept c ON d.drug_concept_id = c.concept_id
        WHERE d.drug_concept_id != 0
          AND d.drug_concept_id NOT IN (
              SELECT DISTINCT descendant_concept_id
              FROM main.concept_ancestor_augmented
              WHERE min_levels_of_separation > 0
          )
        GROUP BY c.vocabulary_id
        ORDER BY n DESC
    """).fetchall()
    print("\n  Remaining missing by vocabulary:")
    for vocab, n in leftover:
        print(f"    {vocab:<25} {n:,} concepts")
    print("\n  Use concept_ancestor_augmented for local analyses; keep Athena concept_ancestor unchanged.")

con.close()
print("\nDone.")
