import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "AbRank_dataset.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "aurlienplissier/abrank",
  file_path,
  pandas_kwargs={
      "sep": "\t" 
  }
)

# ===========================================
# 🧬 AbRank Dataset – Streamlit EDA Dashboard
# ===========================================

st.title('Antibody-Antigen Data Review (EDA)')

# --- Quick info
st.write(f"**Rows:** {df.shape[0]}  |  **Columns:** {df.shape[1]}")
st.write("### 🧾 Column names")
st.write(df.columns.tolist())

st.subheader("🔎 Data Preview")
n_preview = st.slider("Rows to preview", 5, 200, 25, help="Adjust how many rows to show below.")
st.dataframe(df.head(n_preview), use_container_width=True)

# Optional column filter for the preview
with st.expander("Column filter (preview only)"):
    cols_selected = st.multiselect("Select columns to display", df.columns.tolist(), default=df.columns.tolist())
    st.dataframe(df[cols_selected].head(n_preview), use_container_width=True)

# Columns we want numeric -> Coerce to numeric (strings -> NaN)
NUM_COLS = ["Affinity_Kd [nM]", "IC50 [ug/mL]", "log(Kd_ratio)", "log_Aff"]

for col in NUM_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# --- Interactive Filters
st.subheader("🎛️ Interactive Filters")

# Antigen and Antibody selection
antigen_options = sorted(df["Ag_name"].dropna().unique().tolist())
antibody_options = sorted(df["Ab_name"].dropna().unique().tolist())

selected_antigens = st.multiselect("Filter by Antigen", antigen_options, default=[])
selected_antibodies = st.multiselect("Filter by Antibody", antibody_options, default=[])

# Range filters
res_min, res_max = float(df["Affinity_Kd [nM]"].min()), float(df["Affinity_Kd [nM]"].max())
ic50_min, ic50_max = float(df["IC50 [ug/mL]"].min()), float(df["IC50 [ug/mL]"].max())
res_range = st.slider("Affinity (Kd) range [nM]", min_value=res_min, max_value=res_max, value=(res_min, res_max))
ic50_range = st.slider("IC50 range [µg/mL]", min_value=ic50_min, max_value=ic50_max, value=(ic50_min, ic50_max))

# Apply filters
filtered_df = df.copy()
if selected_antigens:
    filtered_df = filtered_df[filtered_df["Ag_name"].isin(selected_antigens)]
if selected_antibodies:
    filtered_df = filtered_df[filtered_df["Ab_name"].isin(selected_antibodies)]
filtered_df = filtered_df[
    (filtered_df["Affinity_Kd [nM]"].between(*res_range)) &
    (filtered_df["IC50 [ug/mL]"].between(*ic50_range))
]

st.caption(f"Filtered rows: **{len(filtered_df)} / {len(df)}**")

# --- Data Preview
st.subheader("🧾 Data Preview")
rows_to_show = st.slider("Rows to display", 5, 100, 20)
st.dataframe(filtered_df.head(rows_to_show), use_container_width=True)

# --- Summary Statistics
st.subheader("📊 Basic Statistics")
num_cols = ["IC50 [ug/mL]", "Affinity_Kd [nM]", "log(Kd_ratio)", "log_Aff"]
st.dataframe(filtered_df[num_cols].describe().T, use_container_width=True)

# --- Visualizations
st.subheader("📈 Visualizations")

# 1️⃣ Distribution of Affinity_Kd
st.markdown("**Distribution of Affinity (Kd [nM])**")
hist_kd = alt.Chart(filtered_df).mark_bar().encode(
    x=alt.X("Affinity_Kd [nM]:Q", bin=alt.Bin(maxbins=40), title="Affinity_Kd [nM]"),
    y="count()",
    tooltip=["count()"]
).properties(height=300)
st.altair_chart(hist_kd, use_container_width=True)

# 2️⃣ Distribution of IC50
st.markdown("**Distribution of IC50 [µg/mL]**")
hist_ic50 = alt.Chart(filtered_df).mark_bar().encode(
    x=alt.X("IC50 [ug/mL]:Q", bin=alt.Bin(maxbins=40), title="IC50 [µg/mL]"),
    y="count()",
    tooltip=["count()"]
).properties(height=300)
st.altair_chart(hist_ic50, use_container_width=True)

# 3️⃣ Antigen Frequency
st.markdown("**Top Antigens by Count**")
antigen_counts = filtered_df["Ag_name"].value_counts().reset_index().rename(columns={"index": "Antigen", "Ag_name": "Count"})
chart_antigens = alt.Chart(antigen_counts.head(20)).mark_bar().encode(
    x="Count:Q",
    y=alt.Y("Antigen:N", sort='-x'),
    tooltip=["Antigen", "Count"]
)
st.altair_chart(chart_antigens, use_container_width=True)

# 4️⃣ Correlation Heatmap
st.markdown("**Correlation Heatmap (numeric columns)**")
corr = filtered_df[num_cols].corr().stack().reset_index()
corr.columns = ["Feature1", "Feature2", "Correlation"]
heat = alt.Chart(corr).mark_rect().encode(
    x="Feature1:N",
    y="Feature2:N",
    color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="redblue", domain=(-1, 1))),
    tooltip=["Feature1", "Feature2", "Correlation"]
).properties(height=350)
st.altair_chart(heat, use_container_width=True)

# 5️⃣ Antigen vs Affinity boxplot
st.markdown("**Affinity distribution by Antigen**")
top_antigens = df["Ag_name"].value_counts().head(15).index.tolist()
box_aff = alt.Chart(filtered_df[filtered_df["Ag_name"].isin(top_antigens)]).mark_boxplot().encode(
    x=alt.X("Ag_name:N", title="Antigen"),
    y=alt.Y("Affinity_Kd [nM]:Q", title="Affinity [nM]"),
    color="Ag_name:N",
    tooltip=["Ag_name", "Affinity_Kd [nM]"]
).properties(height=400)
st.altair_chart(box_aff, use_container_width=True)

# 6️⃣ IC50 vs Affinity scatter
st.markdown("**IC50 vs Affinity (Kd)**")
scatter = alt.Chart(filtered_df).mark_circle(size=60, opacity=0.7).encode(
    x=alt.X("Affinity_Kd [nM]:Q"),
    y=alt.Y("IC50 [ug/mL]:Q"),
    color=alt.Color("Ag_name:N", legend=None),
    tooltip=["Ab_name", "Ag_name", "IC50 [ug/mL]", "Affinity_Kd [nM]"]
).properties(height=400)
st.altair_chart(scatter, use_container_width=True)

# 7️⃣ Sequence Composition (Amino Acid Frequency)
st.subheader("🧪 Sequence Composition")
seq_col = st.selectbox("Select sequence column", ["Ab_heavy_chain_seq", "Ab_light_chain_seq", "Ag_seq"])
aa = list("ACDEFGHIKLMNPQRSTVWY")

def count_aa(seq):
    seq = str(seq).upper()
    return {a: seq.count(a) for a in aa}

seqs = filtered_df[seq_col].dropna().astype(str).tolist()
total_counts = {a: 0 for a in aa}
for s in seqs:
    for a, v in count_aa(s).items():
        total_counts[a] += v
comp_df = pd.DataFrame({"AA": aa, "Count": [total_counts[a] for a in aa]})
comp_df["Frequency"] = comp_df["Count"] / comp_df["Count"].sum()

bar_comp = alt.Chart(comp_df).mark_bar().encode(
    x="AA:N",
    y="Frequency:Q",
    tooltip=["AA", alt.Tooltip("Count:Q"), alt.Tooltip("Frequency:Q", format=".3f")]
).properties(height=300)
st.altair_chart(bar_comp, use_container_width=True)

# 8️⃣ Per-record composition viewer
with st.expander("Per-record composition"):
    chosen_ab = st.selectbox("Select antibody", filtered_df["Ab_name"].unique().tolist())
    row_seq = filtered_df.loc[filtered_df["Ab_name"] == chosen_ab, seq_col].iloc[0]
    row_counts = count_aa(row_seq)
    row_df = pd.DataFrame({"AA": aa, "Count": [row_counts[a] for a in aa]})
    row_df["Frequency"] = row_df["Count"] / row_df["Count"].sum()
    per_chart = alt.Chart(row_df).mark_bar().encode(
        x="AA:N",
        y="Frequency:Q",
        tooltip=["AA", "Count", alt.Tooltip("Frequency:Q", format=".3f")]
    ).properties(height=250)
    st.altair_chart(per_chart, use_container_width=True)

st.caption("🧠 Tip: adjust filters above to explore relationships between affinity, IC50, and antigen properties.")


