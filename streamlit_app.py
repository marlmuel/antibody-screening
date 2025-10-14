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
      "engine": "python",   # needed for sep=None
      "sep": None           # sniff delimiter automatically
  }
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())
st.title('🎈 App Name')

st.write('Hello world!')
