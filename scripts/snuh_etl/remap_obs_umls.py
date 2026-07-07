"""
Strategy F: Native OMOP Mapping + UMLS Metathesaurus API Fallback.

For each remaining unmapped D-code, this script first tries to find a native
OMOP mapping (MeSH -> Maps to -> SNOMED) via the concept_relationship table.
If it fails, it queries the UMLS REST API as a fallback, and finally attempts 
a direct text match.
"""

import os
import urllib.request
import urllib.parse
import json
import duckdb
import urllib.error
from scripts.snuh_etl.etl_utils import backup_database, connect_write

# ── Config ────────────────────────────────────────────────────────────────────
UMLS_API_KEY = os.getenv("UMLS_API_KEY", "YOUR_API_KEY_HERE")
BASE_URL     = "https://uts-ws.nlm.nih.gov/rest"

# Remaining unmapped source values
TARGETS = {
    "HSDN:D057774:Post-Exercise Hypotension":     "D057774",
    "HSDN:D020078:Neurogenic Inflammation":       "D020078",
    "HSDN:D006316:Hearing Loss, High-Frequency":  "D006316",
}

DOMAIN_LIST = "'Observation','Condition','Measurement','Procedure'"

SEP = "-" * 65


def umls_get(path: str, **params) -> dict:
    params["apiKey"] = UMLS_API_KEY
    url = f"{BASE_URL}{path}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        # If the API cannot find atoms, it returns a 404.
        # We catch the error and return an empty dict to continue execution.
        if e.code == 404:
            return {}
        raise e

def get_snomed_atoms(term_name: str) -> list[dict]:
    """Return SNOMED CT atoms for a MeSH descriptor via UMLS search + crosswalk."""
    results = []
    try:
        # Step 1: search by term name to get the UMLS CUI
        data = umls_get(
            "/search/current",
            string=term_name,
            sabs="MSH",
            searchType="words",
            pageSize=5,
        )
        entries = data.get("result", {}).get("results", [])
        if not entries or entries[0].get("ui") == "NONE":
            print(f"  UMLS search: no results for '{term_name}'")
            return results

        cui = entries[0]["ui"]
        print(f"  CUI: {cui}  ({entries[0].get('name', '')})")

        # Step 2: get SNOMED CT preferred terms for that CUI
        data2 = umls_get(
            f"/content/current/CUI/{cui}/atoms",
            sabs="SNOMEDCT_US",
            ttys="PT",
            pageSize=10,
        )
        for atom in data2.get("result", []):
            results.append({
                "name": atom.get("name", ""),
                "code": atom.get("code", ""),
                "cui":  cui,
            })

        # Step 3: if no SNOMED preferred term, try all term types
        if not results:
            data3 = umls_get(
                f"/content/current/CUI/{cui}/atoms",
                sabs="SNOMEDCT_US",
                pageSize=10,
            )
            for atom in data3.get("result", []):
                results.append({
                    "name": atom.get("name", ""),
                    "code": atom.get("code", ""),
                    "cui":  cui,
                })

    except Exception as e:
        print(f"  UMLS error: {e}")
    return results


backup = backup_database("remap_obs_umls")
print(f"Backup created: {backup}")
con = connect_write()
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

total_obs = con.execute("SELECT COUNT(*) FROM main.observation").fetchone()[0]
unmapped_before = con.execute(
    "SELECT COUNT(*) FROM main.observation WHERE observation_concept_id = 0"
).fetchone()[0]
print(f"Before: {unmapped_before:,} unmapped ({unmapped_before/total_obs*100:.1f}%)\n")

total_fixed = 0

for source_value, mesh_id in TARGETS.items():
    n = con.execute("""
        SELECT COUNT(*) FROM main.observation
        WHERE observation_source_value = ? AND observation_concept_id = 0
    """, [source_value]).fetchone()[0]

    if n == 0:
        print(f"[SKIP] already mapped: {source_value}")
        continue

    print(f"\n{SEP}")
    print(f"D-code : {mesh_id}  (n={n:,})")
    print(f"Term   : {source_value}")

    term_name = source_value.split(":", 2)[2] if source_value.startswith("HSDN:") else source_value
    matched = None

    # --- STRATEGY 1: Native OMOP concept_relationship ---
    print("  -> Attempt 1: Native OMOP mapping (concept_relationship)...")
    matched = con.execute("""
        SELECT c2.concept_id, c2.concept_name, c2.domain_id
        FROM main.concept c1
        JOIN main.concept_relationship cr ON c1.concept_id = cr.concept_id_1
        JOIN main.concept c2 ON cr.concept_id_2 = c2.concept_id
        WHERE c1.concept_code = ?
          AND c1.vocabulary_id = 'MeSH'
          AND cr.relationship_id = 'Maps to'
          AND c2.standard_concept = 'S'
          AND c2.invalid_reason IS NULL
        LIMIT 1
    """, [mesh_id]).fetchone()

    # --- STRATEGY 2: UMLS API ---
    if not matched:
        print("  -> OMOP mapping failed. Attempt 2: UMLS API...")
        atoms = get_snomed_atoms(term_name)
        if atoms:
            print(f"  SNOMED atoms from UMLS ({len(atoms)}):")
            for a in atoms:
                print(f"    code={a['code']}  name={a['name']!r}")

            for atom in atoms:
                # Try exact SNOMED concept code match
                row = con.execute(f"""
                    SELECT concept_id, concept_name, domain_id
                    FROM main.concept
                    WHERE concept_code = ?
                      AND vocabulary_id = 'SNOMED'
                      AND standard_concept = 'S'
                      AND invalid_reason IS NULL
                      AND domain_id IN ({DOMAIN_LIST})
                    LIMIT 1
                """, [atom["code"]]).fetchone()

                if not row:
                    # Fallback to exact name match
                    row = con.execute(f"""
                        SELECT concept_id, concept_name, domain_id
                        FROM main.concept
                        WHERE LOWER(concept_name) = LOWER(?)
                          AND standard_concept = 'S'
                          AND invalid_reason IS NULL
                          AND domain_id IN ({DOMAIN_LIST})
                        ORDER BY CASE domain_id
                            WHEN 'Observation' THEN 1
                            WHEN 'Condition'   THEN 2
                            WHEN 'Measurement' THEN 3
                            ELSE 4 END
                        LIMIT 1
                    """, [atom["name"]]).fetchone()

                if row:
                    matched = row
                    break
        else:
            print("  -> No SNOMED atoms found in UMLS")

    # --- STRATEGY 3: Ultimate Text Fallback ---
    if not matched:
        print(f"  -> Attempt 3: Direct text search using original term: '{term_name}'")
        row = con.execute(f"""
            SELECT concept_id, concept_name, domain_id
            FROM main.concept
            WHERE LOWER(concept_name) = LOWER(?)
              AND standard_concept = 'S'
              AND invalid_reason IS NULL
              AND domain_id IN ({DOMAIN_LIST})
            ORDER BY CASE domain_id
                WHEN 'Observation' THEN 1
                WHEN 'Condition'   THEN 2
                WHEN 'Measurement' THEN 3
                ELSE 4 END
            LIMIT 1
        """, [term_name]).fetchone()
        
        if row:
            matched = row

    # --- FINAL CHECK ---
    if not matched:
        print("  -> [TOTAL FAILURE] Could not automatically map this concept.")
        continue

    concept_id, concept_name, domain_id = matched
    print(f"  -> [SUCCESS] Matched: [{domain_id}] '{concept_name}' (id={concept_id})")

    con.execute("""
        INSERT INTO main.etl_observation_remap_audit (
            script_name, observation_source_value, term_name,
            old_observation_concept_id, new_observation_concept_id,
            new_concept_name, vocabulary_id, domain_id, strategy, rows_updated
        )
        VALUES ('remap_obs_umls.py', ?, ?, 0, ?, ?, 'SNOMED', ?, 'umls_or_native', ?)
    """, [source_value, term_name, concept_id, concept_name, domain_id, n])

    con.execute("""
        UPDATE main.observation
        SET observation_concept_id = ?
        WHERE observation_source_value = ?
          AND observation_concept_id   = 0
    """, [concept_id, source_value])
    total_fixed += n
    print(f"  -> Updated {n:,} rows")

unmapped_after = con.execute(
    "SELECT COUNT(*) FROM main.observation WHERE observation_concept_id = 0"
).fetchone()[0]
con.close()

print(f"\n{'='*65}")
print(f"Fixed  : {total_fixed:,} rows")
print(f"Before : {unmapped_before:,} unmapped ({unmapped_before/total_obs*100:.1f}%)")
print(f"After  : {unmapped_after:,} unmapped  ({unmapped_after/total_obs*100:.1f}%)")
print("\nDone.")
