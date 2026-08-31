"""
Remap observation_concept_id = 0 for HSDN-formatted source values.

Source format: 'HSDN:D014972:Xanthoma'
  - HSDN = Human Symptom-Disease Network
  - D014972 = MeSH descriptor ID
  - Xanthoma = human-readable term name

MeSH is not in the Athena standard download, so the D-code cannot be matched
directly. We extract the term name and search SNOMED concepts by:
  A. Exact name match (case-insensitive)
  B. Partial name match (LIKE, for multi-word variations)
  C. Concept synonym match
  D. Hardcoded exact mappings for MeSH inverted-qualifier terms
     (e.g. "Hypotension, Orthostatic" -> "Orthostatic hypotension")
     that the automatic strategies miss due to word-order differences.

Domain policy: OMOP CDM allows storing Condition/Finding concepts in the
observation table — so we accept SNOMED Condition concepts here.
The 'wrong domain' warning in validate_vocabulary.py is a false positive
for sign/symptom observations.
"""

import json
import urllib.parse
import urllib.request

from scripts.snuh_etl.etl_utils import backup_database, connect_write

backup = backup_database("remap_observations")
print(f"Backup created: {backup}")
con = connect_write()

SEP = "-" * 65


def section(title: str):
    print(f"\n{SEP}\n{title}\n{SEP}")


# ── 0. Baseline ──────────────────────────────────────────────────────────────
section("0. BASELINE")
total_obs = con.execute("SELECT COUNT(*) FROM main.observation").fetchone()[0]
unmapped = con.execute(
    "SELECT COUNT(*) FROM main.observation WHERE observation_concept_id = 0"
).fetchone()[0]
print(f"  Total observations : {total_obs:>10,}")
print(f"  Unmapped (cid=0)   : {unmapped:>10,}  ({unmapped/total_obs*100:.1f}%)")

# ── 1. Audit column ───────────────────────────────────────────────────────────
cols = [r[0] for r in con.execute("""
    SELECT column_name FROM information_schema.columns
    WHERE table_schema='main' AND table_name='observation'
""").fetchall()]
if "observation_source_concept_id" not in cols:
    con.execute(
        "ALTER TABLE main.observation ADD COLUMN observation_source_concept_id INTEGER DEFAULT 0"
    )

con.execute("""
    CREATE TABLE IF NOT EXISTS main.etl_observation_remap_audit (
        script_name VARCHAR,
        observation_source_value VARCHAR,
        term_name VARCHAR,
        old_observation_concept_id INTEGER,
        new_observation_concept_id INTEGER,
        new_concept_name VARCHAR,
        vocabulary_id VARCHAR,
        domain_id VARCHAR,
        strategy VARCHAR,
        rows_updated BIGINT,
        mapped_at TIMESTAMP DEFAULT current_timestamp
    )
""")

# ── 2. Extract distinct unmapped HSDN source values ──────────────────────────
section("1. DISTINCT UNMAPPED SOURCE VALUES")
src_rows = con.execute("""
    SELECT observation_source_value, COUNT(*) AS n
    FROM main.observation
    WHERE observation_concept_id = 0
      AND observation_source_value IS NOT NULL
      AND observation_source_value != ''
    GROUP BY observation_source_value
    ORDER BY n DESC
""").fetchall()

print(f"  {len(src_rows)} distinct source values (sample):")
for sv, n in src_rows[:10]:
    print(f"    n={n:>6,}  '{sv}'")

# Parse term names from HSDN:Dxxxxxx:TermName
# Handle both 'HSDN:Dxxxxxx:TermName' and plain strings
def parse_term(source_value: str) -> str:
    """Extract the human-readable term from an HSDN source value."""
    parts = source_value.split(":", 2)
    if len(parts) == 3 and parts[0] == "HSDN":
        return parts[2].strip()
    return source_value.strip()

term_to_sv = {parse_term(sv): sv for sv, _ in src_rows}
print(f"\n  Parsed {len(term_to_sv)} distinct terms")
print("  Sample terms:", list(term_to_sv.keys())[:8])

# ── 3. Build mapping in Python using DuckDB queries per term ────────────────
section("2. SEARCHING CONCEPT TABLE FOR EACH TERM")

# Domains to search in priority order
DOMAIN_PRIORITY = ["Observation", "Condition", "Measurement", "Procedure"]
domain_list = ", ".join(f"'{d}'" for d in DOMAIN_PRIORITY)

# Build a values table: (source_value, term_name)
values_rows = [(sv, parse_term(sv)) for sv, _ in src_rows]

# Create temp table with source values and parsed terms
con.execute("DROP TABLE IF EXISTS main._hsdn_terms")
con.execute("""
    CREATE TABLE main._hsdn_terms (
        source_value VARCHAR,
        term_name    VARCHAR
    )
""")
con.executemany(
    "INSERT INTO main._hsdn_terms VALUES (?, ?)",
    values_rows
)
print(f"  Inserted {len(values_rows)} terms into temp table")

# Strategy A: exact name match (case-insensitive)
print("\n  Strategy A: exact concept_name match...")
con.execute("DROP TABLE IF EXISTS main._obs_mapping")
con.execute(f"""
    CREATE TABLE main._obs_mapping AS
    SELECT DISTINCT
        t.source_value,
        t.term_name,
        c.concept_id,
        c.concept_name,
        c.vocabulary_id,
        c.domain_id,
        'exact_name' AS strategy
    FROM main._hsdn_terms t
    JOIN main.concept c
      ON LOWER(TRIM(t.term_name)) = LOWER(TRIM(c.concept_name))
    WHERE c.standard_concept = 'S'
      AND c.invalid_reason IS NULL
      AND c.domain_id IN ({domain_list})
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY t.source_value
        ORDER BY
            CASE c.domain_id
                WHEN 'Observation' THEN 1
                WHEN 'Condition'   THEN 2
                WHEN 'Measurement' THEN 3
                WHEN 'Procedure'   THEN 4
                ELSE 5
            END,
            LENGTH(c.concept_name),
            c.concept_id
    ) = 1
""")
n_exact = con.execute("SELECT COUNT(*) FROM main._obs_mapping").fetchone()[0]
print(f"    Matched: {n_exact} / {len(values_rows)}")

# Strategy B: partial match (concept_name LIKE '%term%') for remaining
remaining_terms = con.execute("""
    SELECT source_value FROM main._hsdn_terms
    WHERE source_value NOT IN (SELECT source_value FROM main._obs_mapping)
""").fetchall()
print(f"\n  Strategy B: partial concept_name match for {len(remaining_terms)} remaining terms...")

if remaining_terms:
    con.execute(f"""
        INSERT INTO main._obs_mapping
        SELECT DISTINCT
            t.source_value,
            t.term_name,
            c.concept_id,
            c.concept_name,
            c.vocabulary_id,
            c.domain_id,
            'partial_name'
        FROM main._hsdn_terms t
        JOIN main.concept c
          ON LOWER(c.concept_name) LIKE '%' || LOWER(t.term_name) || '%'
        WHERE c.standard_concept = 'S'
          AND c.invalid_reason IS NULL
          AND c.domain_id IN ({domain_list})
          AND t.source_value NOT IN (SELECT source_value FROM main._obs_mapping)
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY t.source_value
            ORDER BY
                CASE c.domain_id
                    WHEN 'Observation' THEN 1
                    WHEN 'Condition'   THEN 2
                    WHEN 'Measurement' THEN 3
                    WHEN 'Procedure'   THEN 4
                    ELSE 5
                END,
                LENGTH(c.concept_name),
                c.concept_id
        ) = 1
    """)
    n_partial = con.execute("SELECT COUNT(*) FROM main._obs_mapping").fetchone()[0] - n_exact
    print(f"    Additional matches: {n_partial}")

# Strategy C: synonym match for remaining
remaining_after_b = con.execute("""
    SELECT COUNT(*) FROM main._hsdn_terms
    WHERE source_value NOT IN (SELECT source_value FROM main._obs_mapping)
""").fetchone()[0]
print(f"\n  Strategy C: synonym match for {remaining_after_b} remaining terms...")

if remaining_after_b > 0:
    con.execute(f"""
        INSERT INTO main._obs_mapping
        SELECT DISTINCT
            t.source_value,
            t.term_name,
            c.concept_id,
            c.concept_name,
            c.vocabulary_id,
            c.domain_id,
            'synonym'
        FROM main._hsdn_terms t
        JOIN main.concept_synonym cs
          ON LOWER(TRIM(t.term_name)) = LOWER(TRIM(cs.concept_synonym_name))
        JOIN main.concept c ON cs.concept_id = c.concept_id
        WHERE c.standard_concept = 'S'
          AND c.invalid_reason IS NULL
          AND c.domain_id IN ({domain_list})
          AND t.source_value NOT IN (SELECT source_value FROM main._obs_mapping)
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY t.source_value
            ORDER BY
                CASE c.domain_id
                    WHEN 'Observation' THEN 1
                    WHEN 'Condition'   THEN 2
                    WHEN 'Measurement' THEN 3
                    WHEN 'Procedure'   THEN 4
                    ELSE 5
                END,
                LENGTH(c.concept_name),
                c.concept_id
        ) = 1
    """)
    n_synonym = con.execute("SELECT COUNT(*) FROM main._obs_mapping").fetchone()[0] - n_exact - (n_partial if remaining_terms else 0)
    print(f"    Additional matches: {n_synonym}")

# ── Strategy D: hardcoded exact mappings for MeSH inverted-qualifier terms ────
# These terms use MeSH format "Qualifier, Main heading" which differs from
# SNOMED word order, so strategies A/B/C miss them despite a valid SNOMED match.
EXACT_MANUAL = {
    "HSDN:D007024:Hypotension, Orthostatic":    319041,   # Orthostatic hypotension
    "HSDN:D006317:Hearing Loss, Noise-Induced": 440422,   # Noise-induced hearing loss
    "HSDN:D020920:Dyssomnias":                  435657,   # Dyssomnia
    "HSDN:D002303:Cardiac Output, Low":         4185565,  # Low cardiac output syndrome
    "HSDN:D010149:Pain, Postoperative":         439502,   # Postoperative pain
    "HSDN:D000648:Amnesia, Retrograde":         4198081,  # Retrograde amnesia
    "HSDN:D017109:Akathisia, Drug-Induced":     4043395,  # Akathisia caused by drug
    "HSDN:D020186:Sleep Bruxism":               433455,   # Sleep related bruxism
}

remaining_d = con.execute("""
    SELECT source_value FROM main._hsdn_terms
    WHERE source_value NOT IN (SELECT source_value FROM main._obs_mapping)
""").fetchall()
remaining_d = [r[0] for r in remaining_d]

manual_inserted = 0
for sv in remaining_d:
    if sv in EXACT_MANUAL:
        cid = EXACT_MANUAL[sv]
        row = con.execute(
            "SELECT concept_name, vocabulary_id, domain_id FROM main.concept WHERE concept_id = ?",
            [cid]
        ).fetchone()
        if row:
            con.execute("""
                INSERT INTO main._obs_mapping
                VALUES (?, ?, ?, ?, ?, ?, 'exact_manual')
            """, [sv, sv.split(":", 2)[2] if sv.startswith("HSDN:") else sv,
                  cid, row[0], row[1], row[2]])
            manual_inserted += 1

n_manual = manual_inserted
print(f"\n  Strategy D (exact manual): {n_manual} additional matches")

# ── Strategy E: MeSH SPARQL entry terms ───────────────────────────────────────
# For each source value still unmapped, fetch all official MeSH entry terms
# (synonyms) via the NLM SPARQL endpoint and try exact + synonym SNOMED match.
MESH_SPARQL = "https://id.nlm.nih.gov/mesh/sparql"

SPARQL_TEMPLATE = """\
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX mesh: <http://id.nlm.nih.gov/mesh/>
PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#>
SELECT ?label WHERE {{
  mesh:{d_code} meshv:concept ?c .
  ?c meshv:term ?t .
  ?t rdfs:label ?label
}}"""


def fetch_mesh_terms(d_code: str) -> list:
    query = SPARQL_TEMPLATE.format(d_code=d_code)
    params = urllib.parse.urlencode({"query": query, "format": "JSON", "limit": "50", "inference": "true"})
    url = f"{MESH_SPARQL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
            return [b["label"]["value"] for b in data["results"]["bindings"]]
    except Exception as e:
        print(f"    WARNING: SPARQL failed for {d_code}: {e}")
        return []


def parse_d_code(source_value: str):
    parts = source_value.split(":", 2)
    return parts[1] if len(parts) >= 2 and parts[0] == "HSDN" else None


def best_snomed_for_term(term: str) -> tuple:
    """Return (concept_id, concept_name, vocabulary_id, domain_id) or None."""
    order = """
        CASE domain_id
            WHEN 'Observation' THEN 1 WHEN 'Condition'   THEN 2
            WHEN 'Measurement' THEN 3 WHEN 'Procedure'   THEN 4 ELSE 5
        END, LENGTH(concept_name)"""

    # exact name
    row = con.execute(f"""
        SELECT concept_id, concept_name, vocabulary_id, domain_id
        FROM main.concept
        WHERE standard_concept = 'S' AND invalid_reason IS NULL
          AND domain_id IN ({domain_list})
          AND LOWER(TRIM(concept_name)) = LOWER(TRIM(?))
        ORDER BY {order} LIMIT 1
    """, [term]).fetchone()
    if row:
        return row

    # exact synonym
    row = con.execute(f"""
        SELECT c.concept_id, c.concept_name, c.vocabulary_id, c.domain_id
        FROM main.concept_synonym cs
        JOIN main.concept c ON cs.concept_id = c.concept_id
        WHERE c.standard_concept = 'S' AND c.invalid_reason IS NULL
          AND c.domain_id IN ({domain_list})
          AND LOWER(TRIM(cs.concept_synonym_name)) = LOWER(TRIM(?))
        ORDER BY {order} LIMIT 1
    """, [term]).fetchone()
    return row


remaining_e = con.execute("""
    SELECT t.source_value, t.term_name
    FROM main._hsdn_terms t
    WHERE t.source_value NOT IN (SELECT source_value FROM main._obs_mapping)
""").fetchall()

n_mesh = 0
if remaining_e:
    print(f"\n  Strategy E: MeSH SPARQL entry terms for {len(remaining_e)} remaining term(s)...")
    for sv, term_name in remaining_e:
        d_code = parse_d_code(sv)
        if not d_code:
            print(f"    {sv}: cannot parse D-code — skipped")
            continue

        mesh_terms = fetch_mesh_terms(d_code)
        if not mesh_terms:
            print(f"    {sv}: no MeSH terms returned")
            continue

        print(f"    {d_code} ({term_name}): {len(mesh_terms)} entry terms fetched")
        matched = None
        matched_via = None
        for mt in mesh_terms:
            hit = best_snomed_for_term(mt)
            if hit:
                matched = hit
                matched_via = mt
                break

        if matched:
            con.execute("""
                INSERT INTO main._obs_mapping VALUES (?, ?, ?, ?, ?, ?, 'mesh_sparql')
            """, [sv, term_name, matched[0], matched[1], matched[2], matched[3]])
            print(f"      -> [{matched[2]}/{matched[3]}] '{matched[1]}'  (via MeSH entry '{matched_via}')")
            n_mesh += 1
        else:
            print(f"      -> no SNOMED match found among {len(mesh_terms)} MeSH terms")
    print(f"    Additional matches via MeSH SPARQL: {n_mesh}")
else:
    print("\n  Strategy E: nothing remaining — skipped")

# ── 4. Show mapping results ────────────────────────────────────────────────────
section("3. MAPPING RESULTS")
total_mapped_sv = con.execute("SELECT COUNT(DISTINCT source_value) FROM main._obs_mapping").fetchone()[0]
print(f"  Mapped {total_mapped_sv} / {len(values_rows)} distinct source values\n")

mapped_rows = con.execute("""
    SELECT source_value, term_name, concept_name, vocabulary_id, domain_id, strategy
    FROM main._obs_mapping
    ORDER BY source_value
""").fetchall()
for _sv, term, cname, vocab, domain, strat in mapped_rows:
    print(f"  '{term}'")
    print(f"    -> [{vocab}/{domain}] '{cname}'  ({strat})")

# Unmapped
unmapped_svs = con.execute("""
    SELECT t.source_value, t.term_name
    FROM main._hsdn_terms t
    WHERE t.source_value NOT IN (SELECT source_value FROM main._obs_mapping)
""").fetchall()
if unmapped_svs:
    print(f"\n  Could NOT map {len(unmapped_svs)} source values:")
    for sv, term in unmapped_svs:
        cnt = next(n for s, n in src_rows if s == sv)
        print(f"    n={cnt:>6,}  term='{term}'")

# ── 5. Apply the mapping ─────────────────────────────────────────────────────
section("4. APPLYING MAPPING")
con.execute("""
    INSERT INTO main.etl_observation_remap_audit (
        script_name, observation_source_value, term_name,
        old_observation_concept_id, new_observation_concept_id,
        new_concept_name, vocabulary_id, domain_id, strategy, rows_updated
    )
    SELECT
        'remap_observations.py',
        o.observation_source_value,
        m.term_name,
        o.observation_concept_id,
        m.concept_id,
        m.concept_name,
        m.vocabulary_id,
        m.domain_id,
        m.strategy,
        COUNT(*)
    FROM main.observation o
    JOIN main._obs_mapping m ON o.observation_source_value = m.source_value
    WHERE o.observation_concept_id = 0
    GROUP BY
        o.observation_source_value, m.term_name, o.observation_concept_id,
        m.concept_id, m.concept_name, m.vocabulary_id, m.domain_id, m.strategy
""")

con.execute("""
    UPDATE main.observation
    SET observation_concept_id = m.concept_id
    FROM main._obs_mapping m
    WHERE main.observation.observation_source_value = m.source_value
      AND main.observation.observation_concept_id = 0
""")

unmapped_after = con.execute(
    "SELECT COUNT(*) FROM main.observation WHERE observation_concept_id = 0"
).fetchone()[0]
mapped_count = unmapped - unmapped_after
print(f"  Rows updated           : {mapped_count:>10,}  ({mapped_count/unmapped*100:.1f}% of previously unmapped)")
print(f"  Still unmapped (cid=0) : {unmapped_after:>10,}  ({unmapped_after/total_obs*100:.1f}% of all observations)")

# ── Cleanup ───────────────────────────────────────────────────────────────────
con.execute("DROP TABLE IF EXISTS main._obs_mapping")
con.execute("DROP TABLE IF EXISTS main._hsdn_terms")
con.close()

print(f"\n{'='*65}")
print("SUMMARY")
print(f"{'='*65}")
print(f"  Before : {unmapped:,} unmapped  ({unmapped/total_obs*100:.1f}%)")
print(f"  After  : {unmapped_after:,} unmapped  ({unmapped_after/total_obs*100:.1f}%)")
print(f"  Fixed  : {mapped_count:,} rows")
print("\nDone.")
