import json

nb = json.load(open('notebooks/t2dm_analysis.ipynb', encoding='utf-8'))

# ── New inventory cell (replaces b981f712) ─────────────────────────────────────
inventory_source = """\
# ── Complete measurement inventory for T2DM patients ──────────────────────────
# Sources: measurement table (lab results + BP) + observation table (vital signs)

# 1. From measurement table
df_meas_lab = con.execute(\"\"\"
    WITH t2 AS (
        SELECT DISTINCT co.person_id FROM main.condition_occurrence co
        JOIN main.concept c ON co.condition_concept_id = c.concept_id
        WHERE c.concept_name ILIKE '%type 2 diabetes mellitus%' AND c.standard_concept = 'S'
    )
    SELECT c.concept_name AS measurement, c.concept_code AS loinc_code,
           'measurement' AS source_table,
           COUNT(*) AS n_records,
           COUNT(DISTINCT m.person_id) AS n_patients,
           ROUND(COUNT(DISTINCT m.person_id)*100.0/(SELECT COUNT(*) FROM t2),1) AS pct_t2dm,
           ROUND(AVG(m.value_as_number),2) AS mean_val,
           ROUND(MIN(m.value_as_number),2) AS min_val,
           ROUND(MAX(m.value_as_number),2) AS max_val,
           m.unit_source_value AS unit
    FROM main.measurement m
    JOIN main.concept c ON m.measurement_concept_id = c.concept_id
    WHERE m.person_id IN (SELECT person_id FROM t2) AND m.measurement_concept_id != 0
    GROUP BY 1,2,3,10 ORDER BY n_records DESC
\"\"\").df()

# 2. Vital signs from observation table (LOINC-coded, numeric)
df_obs_vital = con.execute(\"\"\"
    WITH t2 AS (
        SELECT DISTINCT co.person_id FROM main.condition_occurrence co
        JOIN main.concept c ON co.condition_concept_id = c.concept_id
        WHERE c.concept_name ILIKE '%type 2 diabetes mellitus%' AND c.standard_concept = 'S'
    )
    SELECT c.concept_name AS measurement, c.concept_code AS loinc_code,
           'observation (vital signs)' AS source_table,
           COUNT(*) AS n_records,
           COUNT(DISTINCT o.person_id) AS n_patients,
           ROUND(COUNT(DISTINCT o.person_id)*100.0/(SELECT COUNT(*) FROM t2),1) AS pct_t2dm,
           ROUND(AVG(o.value_as_number),2) AS mean_val,
           ROUND(MIN(o.value_as_number),2) AS min_val,
           ROUND(MAX(o.value_as_number),2) AS max_val,
           o.unit_source_value AS unit
    FROM main.observation o
    JOIN main.concept c ON o.observation_concept_id = c.concept_id
    WHERE o.person_id IN (SELECT person_id FROM t2)
      AND c.vocabulary_id = 'LOINC' AND o.value_as_number IS NOT NULL
    GROUP BY 1,2,3,10 ORDER BY n_records DESC
\"\"\").df()

# Assign categories
def categorise(name):
    n = name.lower()
    if any(k in n for k in ['glucose','hemoglobin a1c','hba1c']): return 'Glycemic'
    if any(k in n for k in ['cholesterol','triglyceride','ldl','hdl']): return 'Lipids'
    if 'blood pressure' in n: return 'Blood pressure'
    if any(k in n for k in ['heart rate','temperature','oxygen','respiratory']): return 'Vital signs'
    return 'Other'

df_meas_lab['category'] = df_meas_lab['measurement'].map(categorise)
df_obs_vital['category'] = df_obs_vital['measurement'].map(categorise)

import pandas as pd
cols = ['measurement','loinc_code','source_table','category',
        'n_records','n_patients','pct_t2dm','mean_val','min_val','max_val','unit']
df_all_meas = pd.concat([df_meas_lab[cols], df_obs_vital[cols]], ignore_index=True)

bmi_present = any(df_all_meas['measurement'].str.contains('bmi|body mass', case=False))
print(f"BMI available in dataset : {bmi_present}  ← explicit limitation")
print(f"Total distinct measurement types : {len(df_all_meas)}")
print()
df_all_meas
"""

# ── Visualisation cell ──────────────────────────────────────────────────────────
viz_source = """\
# ── Visualisation: coverage + value range ─────────────────────────────────────
import matplotlib.patches as mpatches

category_colors = {
    'Glycemic':       '#27ae60',
    'Lipids':          '#2980b9',
    'Blood pressure':  '#8e44ad',
    'Vital signs':     '#e67e22',
    'Other':           '#95a5a6',
}

df_plot = df_all_meas.sort_values('pct_t2dm', ascending=True)
colors  = [category_colors.get(c, '#95a5a6') for c in df_plot['category']]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Measurement Inventory — T2DM Patients (n=3,256)', fontsize=13, fontweight='bold')

# Left: coverage
bars = axes[0].barh(df_plot['measurement'], df_plot['pct_t2dm'], color=colors)
axes[0].set_xlabel('% T2DM patients with ≥1 record')
axes[0].set_title('Coverage by Measurement Type')
axes[0].axvline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
for bar, (_, row) in zip(bars, df_plot.iterrows()):
    axes[0].text(row['pct_t2dm']+0.5, bar.get_y()+bar.get_height()/2,
                 f"{row['pct_t2dm']}%  (n={row['n_patients']:,})",
                 va='center', fontsize=8)
legend_patches = [mpatches.Patch(color=v, label=k) for k,v in category_colors.items()]
axes[0].legend(handles=legend_patches, fontsize=8, loc='lower right')

# Right: mean + range
df_num = df_plot[df_plot['mean_val'].notna()].copy()
colors2 = [category_colors.get(c, '#95a5a6') for c in df_num['category']]
y = range(len(df_num))
axes[1].barh(list(y), df_num['max_val'] - df_num['min_val'],
             left=df_num['min_val'].values, color=colors2, alpha=0.3)
axes[1].scatter(df_num['mean_val'].values, list(y), color=colors2, zorder=5, s=60)
axes[1].set_yticks(list(y))
axes[1].set_yticklabels(
    [f"{row['measurement'][:35]} ({row['unit']})" for _, row in df_num.iterrows()],
    fontsize=8)
axes[1].set_xlabel('Value range (bar) + mean (dot)')
axes[1].set_title('Distribution of Measurement Values')

fig.text(0.5, -0.03,
         '⚠  BMI not available in this dataset — explicit limitation for causal analysis',
         ha='center', fontsize=10, color='tomato', fontweight='bold')

plt.tight_layout()
plt.savefig('t2dm_measurement_inventory.png', dpi=150, bbox_inches='tight')
plt.show()
"""

# Replace cell b981f712
replaced = False
for i, cell in enumerate(nb['cells']):
    if cell.get('id') == 'b981f712':
        nb['cells'][i]['source'] = inventory_source
        nb['cells'][i]['outputs'] = []
        nb['cells'][i]['execution_count'] = None
        # Insert viz cell after
        nb['cells'].insert(i+1, {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': viz_source
        })
        print(f'Replaced b981f712 at [{i}], inserted viz at [{i+1}]')
        replaced = True
        break

if not replaced:
    print('WARNING: cell b981f712 not found')

# Fix remaining wrong LOINC (14743-9 → 1558-6) in all other cells
fixed = 0
for i, cell in enumerate(nb['cells']):
    if cell.get('id') == 'b981f712':
        continue  # already fixed
    src = cell['source']
    if isinstance(src, list):
        joined = ''.join(src)
        if '14743-9' in joined:
            nb['cells'][i]['source'] = joined.replace('14743-9', '1558-6').splitlines(keepends=True)
            fixed += 1
            print(f'  Fixed LOINC in cell [{i}] id={cell.get("id")}')
    elif isinstance(src, str) and '14743-9' in src:
        nb['cells'][i]['source'] = src.replace('14743-9', '1558-6')
        fixed += 1
        print(f'  Fixed LOINC in cell [{i}] id={cell.get("id")}')

print(f'Fixed {fixed} other cells with wrong LOINC')

with open('notebooks/t2dm_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f'Done. Total cells: {len(nb["cells"])}')
