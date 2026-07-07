from scripts.snuh_etl.etl_utils import backup_database, connect_write, sql_literal_path, vocab_dir


VOC_DIR = vocab_dir()

# Athena files are TAB-separated despite the .csv extension.
TABLES = {
    "concept": "CONCEPT.csv",
    "concept_relationship": "CONCEPT_RELATIONSHIP.csv",
    "concept_ancestor": "CONCEPT_ANCESTOR.csv",
    "vocabulary": "VOCABULARY.csv",
    "concept_synonym": "CONCEPT_SYNONYM.csv",
    "drug_strength": "DRUG_STRENGTH.csv",
    "domain": "DOMAIN.csv",
    "concept_class": "CONCEPT_CLASS.csv",
    "relationship": "RELATIONSHIP.csv",
}

REQUIRED = {"concept", "concept_relationship", "concept_ancestor", "vocabulary"}

missing_required = [
    filename
    for table, filename in TABLES.items()
    if table in REQUIRED and not (VOC_DIR / filename).exists()
]
if missing_required:
    raise FileNotFoundError(
        "Missing required Athena vocabulary file(s): "
        + ", ".join(missing_required)
        + f" in {VOC_DIR}"
    )

backup = backup_database("load_vocabulary")
print(f"Backup created: {backup}")

con = connect_write()

try:
    con.execute("BEGIN TRANSACTION")

    for table, filename in TABLES.items():
        path = VOC_DIR / filename
        if not path.exists():
            print(f"SKIP  {filename} (optional file not found)")
            continue

        print(f"Loading {filename:<40} -> {table} ...", end=" ", flush=True)
        con.execute(f'DROP TABLE IF EXISTS main."{table}"')
        con.execute(f"""
            CREATE TABLE main."{table}" AS
            SELECT * FROM read_csv(
                '{sql_literal_path(path)}',
                delim = '\t',
                header = true,
                quote = '',
                all_varchar = false
            )
        """)
        n = con.execute(f'SELECT COUNT(*) FROM main."{table}"').fetchone()[0]
        print(f"{n:>12,} rows  OK")

    con.execute("COMMIT")
except Exception:
    con.execute("ROLLBACK")
    raise

print()
print("Verifying FastOMOP required tables:")
for t in sorted(REQUIRED):
    try:
        n = con.execute(f'SELECT COUNT(*) FROM main."{t}"').fetchone()[0]
        print(f"  OK      {t:<30} {n:>12,} rows")
    except Exception as e:
        print(f"  MISSING {t:<30} ({e})")

con.close()
print()
print("Done.")
