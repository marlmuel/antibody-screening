import kagglehub
from kagglehub import KaggleDatasetAdapter
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly
import plotly.express as px
import plotly.figure_factory as ff

# Data ingestion

example_note = st.expander("Column expectations (click to expand)")
with example_note:
  st.markdown(
  """
  Expected columns (case-sensitive; extra columns are fine):


  - `Ab_name`, `Ag_name`, `Ag_name_details`
  - `IC50 [ug/mL]`, `Affinity_Kd [nM]`, `log(Kd_ratio)`, `log_Aff`
  - `Ag_epitope_restrictions`, `Source`, `escape`, `Aff_op`
  - `Ab_heavy_chain_seq`, `Ab_light_chain_seq`, `Ag_seq`
  - Structure meta: `Ab_structure_method`, `Ag_structure_method`, `bound_AbAg_structure_method`
  - PDB ids: `Ab_PDB_ID`, `Ag_PDB_ID`, `bound_AbAg_PDB_ID`
  - Cluster labels: `Ab_Lev3_cluster`, `Ab10_cluster`, `Ab25_cluster`, `Ab50_cluster`,
  `Ag_Lev3_cluster`, `Ag10_cluster`, `Ag25_cluster`, `Ag50_cluster`
  """
  )

# @st.cache_data(show_spinner=False)
def read_data():
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
  # downsample to 2% if too large
  #if len(df) > 1000:
  #  frac = 0.02
  #  st.warning(f"Dataset has {len(df):,} rows — sampling {frac*100:.0f}% for performance.")
  #  df = df.sample(frac=frac, random_state=42)
  return df


# -----------------------------
# NA Analysis
# -----------------------------
def remove_na(df: pd.DataFrame):
  pct_na = df.isna().sum() / len(df) * 100
  st.write(pct_na)
  df = df.loc[:, (df.isnull().sum(axis=0) <= pct_na)]
  st.write(df.columns)
  return(df)


# ===========================================
# 🧬 AbRank Dataset – Streamlit EDA Dashboard
# ===========================================

st.set_page_config(
    page_title="Antibody–Antigen EDA",
    layout="wide",
)

st.title('Antibody-Antigen Data Review (EDA)')

# @st.cache_data(show_spinner=False)
def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Numeric columns we care about
    num_cols = [
        "IC50 [ug/mL]",
        "Affinity_Kd [nM]",
        "log(Kd_ratio)",
        "log_Aff",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("NA", "", regex=False)
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Quick sequence lengths (may help for proxy features)
    for c in ["Ab_heavy_chain_seq", "Ab_light_chain_seq", "Ag_seq"]:
        if c in df.columns:
            df[f"{c}_len"] = df[c].astype(str).str.len()

    # Simple heuristic CDR3-ish length proxy (look for motif starting with 'CAR' and ending with 'WGQG' or 'WGGG' etc.)
    if "Ab_heavy_chain_seq" in df.columns:
        def cdrh3_len(seq: str):
            if not isinstance(seq, str):
                return np.nan
            # Very rough regex; not IMGT-accurate, but helpful for EDA
            m = re.search(r"C[A-Z]{2,30}W(GQG|GGG|GXG)", seq)
            return len(m.group(0)) if m else np.nan
        df["CDRH3_len_proxy"] = df["Ab_heavy_chain_seq"].apply(cdrh3_len)

    return df


def add_feature_notes():
    st.info(
        "Applied cleaning: coerced numeric fields, created *_len columns, and a rough CDRH3 length proxy.",
        icon="ℹ️",
    )


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    cat_cols = [
        "Ag_name", "Ag_name_details", "Source", "Ab_structure_method",
        "Ag_structure_method", "bound_AbAg_structure_method",
    ]
    for c in [cc for cc in cat_cols if cc in df.columns]:
        vals = ["(all)"] + sorted([v for v in df[c].dropna().astype(str).unique()])
        choice = st.sidebar.selectbox(c, vals)
        if choice != "(all)":
            df = df[df[c].astype(str) == choice]

    # Range filters
    if "Affinity_Kd [nM]" in df.columns:
        min_v, max_v = float(df["Affinity_Kd [nM]"].min()), float(df["Affinity_Kd [nM]"].max())
        if not np.isnan(min_v) and not np.isnan(max_v) and min_v < max_v:
            r = st.sidebar.slider("Affinity_Kd [nM] range", min_v, max_v, (min_v, max_v))
            df = df[df["Affinity_Kd [nM]"].between(*r)]

    if "IC50 [ug/mL]" in df.columns:
        min_v, max_v = float(df["IC50 [ug/mL]"].min()), float(df["IC50 [ug/mL]"].max())
        if not np.isnan(min_v) and not np.isnan(max_v) and min_v < max_v:
            r = st.sidebar.slider("IC50 [ug/mL] range", min_v, max_v, (min_v, max_v))
            df = df[df["IC50 [ug/mL]"].between(*r)]

    search_str = st.sidebar.text_input("Search Ab/Ag name contains…")
    if search_str:
        mask = pd.Series(False, index=df.index)
        for c in ["Ab_name", "Ag_name", "Ag_name_details"]:
            if c in df.columns:
                mask = mask | df[c].astype(str).str.contains(search_str, case=False, na=False)
        df = df[mask]
    return df


def kpi_cards(df: pd.DataFrame):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Rows", f"{len(df):,}")
    with c2:
        st.metric("Unique Abs", df["Ab_name"].nunique() if "Ab_name" in df.columns else 0)
    with c3:
        st.metric("Unique Ags", df["Ag_name"].nunique() if "Ag_name" in df.columns else 0)
    with c4:
        n_pdb = 0
        for c in ["Ab_PDB_ID","Ag_PDB_ID","bound_AbAg_PDB_ID"]:
            if c in df.columns:
                n_pdb += df[c].notna().sum()
        st.metric("PDB IDs (non-null)", int(n_pdb))
    with c5:
        na_share = df.isna().mean().mean() if len(df) else 0.0
        st.metric("Overall NA %", f"{na_share*100:.1f}%")


def distribution_plots(df: pd.DataFrame):
    st.subheader("Distributions")
    cols = []
    if "Affinity_Kd [nM]" in df.columns:
        cols.append("Affinity_Kd [nM]")
    if "IC50 [ug/mL]" in df.columns:
        cols.append("IC50 [ug/mL]")
    if "log(Kd_ratio)" in df.columns:
        cols.append("log(Kd_ratio)")
    if "log_Aff" in df.columns:
        cols.append("log_Aff")

    if cols:
        choice = st.selectbox("Metric", cols)
        group = st.selectbox("Group by (optional)", ["(none)"] + [c for c in ["Ag_name","Ab_name","Ag_Lev3_cluster","Ab_Lev3_cluster","Source"] if c in df.columns])
        data = df.dropna(subset=[choice])
        if group != "(none)" and group in df.columns:
            fig = px.violin(data, x=group, y=choice, box=True, points=False)
        else:
            fig = px.histogram(data, x=choice, nbins=40, marginal="box")
        st.plotly_chart(fig, use_container_width=True)


def scatter_plots(df: pd.DataFrame):
    st.subheader("Affinity vs IC50")
    if not set(["Affinity_Kd [nM]","IC50 [ug/mL]"]).issubset(df.columns):
        st.info("Both Affinity_Kd [nM] and IC50 [ug/mL] are needed for this plot.")
        return
    data = df.dropna(subset=["Affinity_Kd [nM]","IC50 [ug/mL]"]).copy()
    if data.empty:
        st.info("No overlapping non-null values to plot.")
        return
    data["Affinity_Kd_log10"] = np.log10(data["Affinity_Kd [nM]"])
    data["IC50_log10"] = np.log10(data["IC50 [ug/mL]"])

    color_col = "Ag_name" if "Ag_name" in data.columns else None
    fig = px.scatter(
        data, x="Affinity_Kd_log10", y="IC50_log10",
        color=color_col, hover_data=[c for c in ["Ab_name","Ag_name_details","Ab_PDB_ID","Ag_PDB_ID"] if c in data.columns]
    )
    fig.update_layout(xaxis_title="log10(Kd [nM])", yaxis_title="log10(IC50 [µg/mL])")
    st.plotly_chart(fig, use_container_width=True)


def heatmap(df: pd.DataFrame):
    st.subheader("Median affinity by Ab/Ag cluster")
    needed = ["Ab_Lev3_cluster","Ag_Lev3_cluster","Affinity_Kd [nM]"]
    if not set(needed).issubset(df.columns):
        st.info("Need Ab_Lev3_cluster, Ag_Lev3_cluster, Affinity_Kd [nM] for this heatmap.")
        return
    pvt = df.pivot_table(index="Ag_Lev3_cluster", columns="Ab_Lev3_cluster", values="Affinity_Kd [nM]", aggfunc="median")
    if pvt.empty:
        st.info("Nothing to aggregate.")
        return
    fig = ff.create_annotated_heatmap(
        z=pvt.values,
        x=pvt.columns.astype(str).tolist(),
        y=pvt.index.astype(str).tolist(),
        colorscale="Viridis",
        showscale=True,
        annotation_text=np.round(pvt.values, 2),
    )
    st.plotly_chart(fig, use_container_width=True)


def structure_counts(df: pd.DataFrame):
    st.subheader("Structure method counts")
    cols = [c for c in ["Ab_structure_method","Ag_structure_method","bound_AbAg_structure_method"] if c in df.columns]
    if not cols:
        st.info("No structure method columns present.")
        return
    counts = (
        df.melt(value_vars=cols, var_name="which", value_name="method")
          .dropna(subset=["method"]) 
          .groupby(["which","method"]) 
          .size() 
          .reset_index(name="count")
    )
    fig = px.bar(counts, x="method", y="count", color="which", barmode="group")
    st.plotly_chart(fig, use_container_width=True)


def sequence_lengths(df: pd.DataFrame):
    st.subheader("Sequence length relationships")
    numeric = [c for c in ["Ab_heavy_chain_seq_len","Ab_light_chain_seq_len","Ag_seq_len","CDRH3_len_proxy","Affinity_Kd [nM]","IC50 [ug/mL]"] if c in df.columns]
    opts_x = [c for c in numeric if c not in ("Affinity_Kd [nM]","IC50 [ug/mL]")]
    if not opts_x:
        st.info("Upload sequences to enable length-based plots.")
        return
    x = st.selectbox("X axis", opts_x)
    y = st.selectbox("Y axis", [c for c in ["Affinity_Kd [nM]","IC50 [ug/mL]","log(Kd_ratio)","log_Aff"] if c in df.columns])
    data = df.dropna(subset=[x, y])
    if data.empty:
        st.info("No data after dropping NAs for selected axes.")
        return
    fig = px.scatter(data, x=x, y=y, color="Ag_name" if "Ag_name" in df.columns else None, trendline="ols")
    st.plotly_chart(fig, use_container_width=True)


def table(df: pd.DataFrame):
    st.subheader("Data table")
    st.dataframe(df, use_container_width=True)


# -----------------------------
# Main app flow
# -----------------------------

raw = read_data()
st.write(raw.head())
na_removed = remove_na(raw)
clean = coerce_types(na_removed)
add_feature_notes()

filt = sidebar_filters(clean)
kpi_cards(filt)

col1, col2 = st.columns([1.2, 1])
with col1:
    distribution_plots(filt)
with col2:
    structure_counts(filt)

scatter_plots(filt)
heatmap(filt)
#sequence_lengths(filt)
table(filt)
