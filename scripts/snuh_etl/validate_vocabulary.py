import duckdb
from scripts.snuh_etl.etl_utils import db_path

DB_PATH = db_path()
con = duckdb.connect(str(DB_PATH), read_only=True)

SEP = "-" * 65

def section(title):
    print()
    print(SEP)
    print(title)
    print(SEP)

def pct(n, total):
    return f"{n/total*100:.1f}%" if total > 0 else "N/A"

# ── 1. Required tables presence ───────────────────────────────────────────────
section("1. REQUIRED TABLES")
required = [
    "person", "visit_occurrence", "condition_occurrence",
    "drug_exposure", "measurement", "observation",
    "procedure_occurrence", "death",
    "concept", "concept_relationship", "concept_ancestor", "vocabulary",
]
all_present = True
for t in required:
    try:
        n = con.execute(f'SELECT COUNT(*) FROM main."{t}"').fetchone()[0]
        print(f"  OK      {t:<35} {n:>12,} rows")
    except Exception:
        print(f"  MISSING {t}")
        all_present = False

print()
print(f"  {'All required tables present' if all_present else 'WARNING: some tables missing'}")

# ── 2. Concept coverage per clinical table ────────────────────────────────────
section("2. CONCEPT COVERAGE (concept_id found in concept table)")

checks = [
    ("condition_occurrence", "condition_concept_id",  "Condition"),
    ("drug_exposure",        "drug_concept_id",        "Drug"),
    ("measurement",          "measurement_concept_id", "Measurement"),
    ("observation",          "observation_concept_id", "Observation"),
    ("visit_occurrence",     "visit_concept_id",       "Visit"),
]

allowed_domains = {
    "observation": ("Observation", "Condition", "Measurement", "Procedure"),
}

for table, col, domain in checks:
    try:
        total = con.execute(f"SELECT COUNT(*) FROM main.{table}").fetchone()[0]
        nonzero = con.execute(
            f"SELECT COUNT(*) FROM main.{table} WHERE {col} != 0"
        ).fetchone()[0]
        matched = con.execute(f"""
            SELECT COUNT(*) FROM main.{table} t
            JOIN main.concept c ON t.{col} = c.concept_id
            WHERE t.{col} != 0
        """).fetchone()[0]
        std = con.execute(f"""
            SELECT COUNT(*) FROM main.{table} t
            JOIN main.concept c ON t.{col} = c.concept_id
            WHERE t.{col} != 0 AND c.standard_concept = 'S'
        """).fetchone()[0]
        expected_domains = allowed_domains.get(table, (domain,))
        expected_sql = ",".join(f"'{d}'" for d in expected_domains)
        wrong_domain = con.execute(f"""
            SELECT COUNT(*) FROM main.{table} t
            JOIN main.concept c ON t.{col} = c.concept_id
            WHERE t.{col} != 0
              AND c.standard_concept = 'S'
              AND c.domain_id NOT IN ({expected_sql})
        """).fetchone()[0]

        print(f"\n  {table} ({col})")
        print(f"    Total rows          : {total:>10,}")
        print(f"    Non-zero concept_id : {nonzero:>10,}  ({pct(nonzero, total)})")
        print(f"    Found in concept    : {matched:>10,}  ({pct(matched, nonzero)} of non-zero)")
        print(f"    Standard (S)        : {std:>10,}  ({pct(std, matched)} of found)")
        if wrong_domain > 0:
            print(f"    Wrong domain        : {wrong_domain:>10,}  WARNING - expected {expected_domains}")
        else:
            print(f"    Domain {expected_domains} check  : OK")
    except Exception as e:
        print(f"  ERROR on {table}: {e}")

# ── 3. Unmapped concepts (concept_id = 0) ─────────────────────────────────────
section("3. UNMAPPED CONCEPTS (concept_id = 0)")
for table, col, _ in checks:
    try:
        total = con.execute(f"SELECT COUNT(*) FROM main.{table}").fetchone()[0]
        zero  = con.execute(
            f"SELECT COUNT(*) FROM main.{table} WHERE {col} = 0"
        ).fetchone()[0]
        flag = "  WARNING" if zero / total > 0.1 else ""
        print(f"  {table:<30} {zero:>8,} / {total:>10,}  ({pct(zero, total)}){flag}")
    except Exception as e:
        print(f"  ERROR on {table}: {e}")

# ── 4. Concept ancestor coverage ─────────────────────────────────────────────
section("4. CONCEPT_ANCESTOR COVERAGE")
ancestor_table = "concept_ancestor"
try:
    con.execute("SELECT 1 FROM main.concept_ancestor_augmented LIMIT 1")
    ancestor_table = "concept_ancestor_augmented"
except Exception:
    pass

if ancestor_table == "concept_ancestor_augmented":
    print("  Using main.concept_ancestor_augmented (Athena + local extension)")
else:
    print("  Using main.concept_ancestor")

for table, col, _ in checks[:3]:  # condition, drug, measurement only
    try:
        distinct_ids = con.execute(
            f"SELECT COUNT(DISTINCT {col}) FROM main.{table} WHERE {col} != 0"
        ).fetchone()[0]
        in_ancestor = con.execute(f"""
            SELECT COUNT(DISTINCT t.{col})
            FROM main.{table} t
            JOIN main.{ancestor_table} ca ON t.{col} = ca.descendant_concept_id
            WHERE t.{col} != 0
        """).fetchone()[0]
        true_parent = con.execute(f"""
            SELECT COUNT(DISTINCT t.{col})
            FROM main.{table} t
            JOIN main.{ancestor_table} ca ON t.{col} = ca.descendant_concept_id
            WHERE t.{col} != 0
              AND ca.min_levels_of_separation > 0
        """).fetchone()[0]
        print(f"  {table:<30} {in_ancestor}/{distinct_ids} concept_ids in {ancestor_table}, incl. self ({pct(in_ancestor, distinct_ids)})")
        print(f"  {'':<30} {true_parent}/{distinct_ids} with non-self parent coverage ({pct(true_parent, distinct_ids)})")
    except Exception as e:
        print(f"  ERROR on {table}: {e}")

# ── 5. Vocabulary breakdown of matched concepts ────────────────────────────────
section("5. VOCABULARY BREAKDOWN (which vocabs cover clinical concepts)")
for table, col, _ in checks[:3]:
    try:
        rows = con.execute(f"""
            SELECT c.vocabulary_id, c.standard_concept, COUNT(*) as n
            FROM main.{table} t
            JOIN main.concept c ON t.{col} = c.concept_id
            WHERE t.{col} != 0
            GROUP BY c.vocabulary_id, c.standard_concept
            ORDER BY n DESC
            LIMIT 8
        """).fetchall()
        print(f"\n  {table}:")
        for vocab, std, n in rows:
            print(f"    {vocab:<25} standard={std}  {n:>8,}")
    except Exception as e:
        print(f"  ERROR on {table}: {e}")

# ── 6. Source value to concept mapping check ──────────────────────────────────
section("6. SOURCE VALUE -> CONCEPT MAPPING SAMPLE")
for table, col, _ in [("condition_occurrence","condition_concept_id",""), ("drug_exposure","drug_concept_id","")]:
    src_col = col.replace("_concept_id", "_source_value")
    try:
        rows = con.execute(f"""
            SELECT t.{src_col}, c.concept_name, c.vocabulary_id, c.standard_concept
            FROM main.{table} t
            JOIN main.concept c ON t.{col} = c.concept_id
            WHERE t.{col} != 0
            GROUP BY t.{src_col}, c.concept_name, c.vocabulary_id, c.standard_concept
            LIMIT 5
        """).fetchall()
        print(f"\n  {table}:")
        for src, name, vocab, std in rows:
            print(f"    source='{src}'  -->  '{name}' [{vocab}, std={std}]")
    except Exception as e:
        print(f"  ERROR on {table}: {e}")

# ── 7. FastOMOP readiness summary ─────────────────────────────────────────────
section("7. FASTOMOP READINESS SUMMARY")
checks_summary = []

# Schema
schemas = [r[0] for r in con.execute(
    "SELECT DISTINCT table_schema FROM information_schema.tables"
).fetchall()]
checks_summary.append(("Schema is 'main'", "main" in schemas))

# Required vocab tables
for t in ["concept", "concept_relationship", "concept_ancestor", "vocabulary"]:
    try:
        n = con.execute(f'SELECT COUNT(*) FROM main."{t}"').fetchone()[0]
        checks_summary.append((f"Table '{t}' populated", n > 0))
    except Exception:
        checks_summary.append((f"Table '{t}' populated", False))

# Concept coverage > 80%
try:
    nonzero = con.execute("SELECT COUNT(*) FROM main.condition_occurrence WHERE condition_concept_id != 0").fetchone()[0]
    matched = con.execute("""
        SELECT COUNT(*) FROM main.condition_occurrence t
        JOIN main.concept c ON t.condition_concept_id = c.concept_id
        WHERE t.condition_concept_id != 0
    """).fetchone()[0]
    checks_summary.append(("Condition concept coverage >= 80%", (matched/nonzero >= 0.8) if nonzero > 0 else False))
except Exception:
    checks_summary.append(("Condition concept coverage >= 80%", False))

for label, ok in checks_summary:
    status = "OK  " if ok else "FAIL"
    print(f"  [{status}] {label}")

con.close()
print()
print("Validation complete.")
