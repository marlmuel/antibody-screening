# 🧬 Antibody Screening App

This app provides interactive analysis of publicly available antibody screening data from Hugging Face. It enables researchers and scientists to explore antibody/antigen interactions, assay results, and perform sequence analysis.

## Live Demo

[📊 View the App](https://antibody-screening-oc4r4xemxvxxbqxovgaoo6.streamlit.app/)

---

## Data Overview

### Data Source
The app uses publicly available antibody screening datasets from Hugging Face, containing comprehensive information about:
- **Antibodies**: Protein sequences and structural information
- **Antigens**: Target molecules for antibody binding
- **Assay Results**: Quantitative measurements of antibody-antigen interactions

### Data Fields & Meaning
- **Binding Kinetics**: Measures the rate and affinity of antibody-antigen binding (e.g., Kd values, on-rate, off-rate)
- **Amino Acid (AA) Sequences**: The protein sequence composition of antibodies and antigens
- **Assay Types**: Different experimental methods used to measure binding (e.g., ELISA, Surface Plasmon Resonance)
- **Affinity Data**: Quantitative binding strength measurements between antibody and antigen pairs

---

## Report Features

### Data Processing Pipeline
The app performs the following analysis steps:

1. **Data Cleaning**
   - Validation of sequence formats
   - Removal of incomplete records
   - Standardization of assay measurements
   - Handling of outliers and missing values

2. **Data Visualization**
   - Scatter plots showing relationships between:
     - Binding affinity vs. sequence length
     - Assay results vs. experimental conditions
     - Multiple antibody-antigen interaction metrics

3. **Sequence Analysis**
   - Multiple sequence alignment (MSA)
   - Conservation analysis across antibody variants
   - Identification of key binding regions

### Filtering & Results
Users can filter the dataset by:
- **Affinity Range**: Set minimum/maximum binding strength thresholds
- **Sequence Length**: Filter by protein size
- **Assay Type**: Select specific experimental methods
- **Antigen Target**: Filter by specific target molecules

**Output**: Filtered results are displayed as:
- Interactive data tables with sortable columns
- Visual plots showing filtered subset patterns
- Downloadable alignment files for selected sequences
- Summary statistics of filtered antibodies/antigens

---

## Getting Started

### Prerequisites
- Python 3.8+
- Streamlit
- Required dependencies (see requirements.txt)

### Installation
```bash
git clone https://github.com/marlmuel/antibody-screening.git
cd antibody-screening
pip install -r requirements.txt
```

### Running Locally
```bash
streamlit run app.py
```

---

## Further Reading

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Antibody-Antigen Binding Kinetics](https://en.wikipedia.org/wiki/Antibody)
- [Sequence Alignment Techniques](https://en.wikipedia.org/wiki/Sequence_alignment)
- [Hugging Face Datasets](https://huggingface.co/datasets)
