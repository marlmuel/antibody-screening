import streamlit as st
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

st.title('Antibody-Antigen Data Review (EDA)')

st.write(df.columns.tolist())

st.subheader("🔎 Data Preview")
n_preview = st.slider("Rows to preview", 5, 200, 25, help="Adjust how many rows to show below.")
st.dataframe(df.head(n_preview), use_container_width=True)

# Optional column filter for the preview
with st.expander("Column filter (preview only)"):
    cols_selected = st.multiselect("Select columns to display", df.columns.tolist(), default=df.columns.tolist())
    st.dataframe(df[cols_selected].head(n_preview), use_container_width=True)

