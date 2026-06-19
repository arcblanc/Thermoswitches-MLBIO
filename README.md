# Thermoswitches-ML


# 🧬 RNA Thermoswitch Engineering: A Machine Learning & Biophysics Pipeline

**Author:** [Amier Zuhri]  
**Role:** Biotech Machine Learning Engineer  
**Domain:** Synthetic Biology, Bioinformatics, Machine Learning  

## 📌 Project Overview
This project aims to computationally classify and engineer **prokaryotic RNA thermoswitches** (RNA thermometers). Biologically, these are highly structured *cis*-regulatory non-coding RNA elements (often located in the 5' UTR) that fold into hairpins at low temperatures to physically sequester the Shine-Dalgarno (SD) sequence and prevent ribosome binding. Upon a temperature increase, the hairpin melts in a zipper-like fashion, exposing the SD sequence and activating translation. 

**The Engineering Goal:** Build a robust machine learning pipeline(Random Forest) to classify highly functional thermoswitches, with the ultimate future goal of utilizing deep learning foundation models to engineer a synthetic thermoswitch that triggers precisely at 55°C with zero low-temperature "leakiness."

---

## 🏗️ Architecture & MLOps
Following modern MLOps architectures, this repository is structured as a reusable Python package. 
*   **Centralized Data Logic:** All extraction scripts are unified within a `data.py` module.
*   **Experiment Tracking:** Model parameters, metrics, and serialized artifacts are tracked and logged using **MLflow**.
*   **CLI Automation:** A `Makefile` is used to automate command-line directives (e.g., `make install`, `make clean`).

---

## 🛠️ Phase 1: Data Extraction & Balancing Workflow
To prevent the model from learning evolutionary background noise or falling victim to class imbalance, we employ a strict, multi-step data processing pipeline:

1. **Broad SQL Extraction (Rfam):** 
   *   **Positives:** ~2,960 known prokaryotic, heat-shock thermoswitches (e.g., ROSE, FourU elements) extracted from the Rfam relational database.
   *   **Negatives:** A massive pool (~168,000) of standard, non-temperature-responsive bacterial 5' UTRs and *Cis*-regulatory elements from the same biological neighborhood. 
2. **FASTA Retrieval:** We utilize Biopython's `Entrez` module to fetch the raw nucleotide sequences (A, C, G, U) using the genomic coordinates (`seq_start`, `seq_end`) from Rfam.
3. **Homology Filtering (CD-HIT):** *Before* balancing, both positive and negative FASTA datasets are passed independently through the **CD-HIT algorithm**. By setting a sequence identity threshold (e.g., 25%), we remove highly homologous and redundant sequences that would otherwise cause severe data leakage and overfitting.
4. **Taxonomic Stratified Undersampling:** To create a perfectly balanced 1:1 training dataset, we use Pandas to undersample the surviving negative pool. We stratify this sampling to perfectly mimic the evolutionary lineage distribution (~126 order-level groups) of the positive dataset, neutralizing evolutionary bias.

---

## 🌡️ Phase 2: Biophysical Feature Engineering
Static nucleotide strings cannot easily convey temperature dynamics to classical algorithms. We translate the thermodynamics of the RNA into mathematical features:
1. **Simulation:** The consolidated dataset is run through **NuPACK**, a computational suite for analyzing nucleic acid thermodynamics. We simulate test tube ensembles across a temperature gradient (20°C to 70°C) to calculate the Ribosome Binding Site (RBS) exposure.
2. **Hill Function Fitting:** The simulated melting data is fitted to a logistic sigmoid curve (Hill Function) to extract three continuous, non-linear features for our classifier:
   *   **Midpoint ($T_m$):** The inflection point determining the activation temperature.
   *   **Hill Coefficient (Slope):** The steepness of the curve, defining how "zipper-like" or digital the switch is.
   *   **Amplitude:** The difference between the top and bottom asymptotes, representing the strength of the switch and identifying low-temperature leakiness.

---

## 🤖 Phase 3: Machine Learning Classification
To classify functional switches based on our engineered features, we employ an ensemble learning approach:
*   **The Algorithm:** **Random Forest Classifier**.
*   **Why Random Forest?** A single decision tree breaks down data via binary decisions to maximize Information Gain, but it suffers from high variance and is highly prone to overfitting. Random Forest solves this by building a massive ensemble of decision trees, each trained on a bootstrapped subset of the data and restricted to a random subset of features.
*   **Output:** The forest aggregates all individual tree predictions and outputs a **Majority Vote**, yielding a robust, generalizable classifier that resists overfitting.

---

## 🚀 Phase 4: Future Plans (Deep Learning & Foundation Models)
While Random Forest is excellent for *classifying* existing features, our future roadmap transitions to deep learning to *engineer* novel sequences targeting exactly 55°C.

*   **Leveraging RNA Foundation Models:** We potentially plan to integrate large, pre-trained transformer models like **BiRNA-BERT** (117M parameters) and **RiNALMo** (650M parameters). These models have been pre-trained on tens of millions of unannotated non-coding RNAs from RNAcentral.
*   **Alpha Fold:** We will explore Google's Alpha Fold to maybe input non euclidean RNA information 
*   **Inverse Folding:** By fine-tuning these models on our carefully curated dataset, the Transformers will learn the hidden sequence-to-structure grammar directly from the raw nucleotides. This will allow the algorithm to suggest the precise base-pair mutations required to engineer a leak-free, 55°C-activated thermoswitch. 

---

## 💻 Setup & Installation
*(Standard placeholder for your repo instructions)*

```bash
# 1. Clone the repository
git clone https://github.com/arcblanc/Thermoswitches-MLBIO.git
cd RNA-Thermoswitch-Classifier

# 2. Create and activate a virtual environment
pyenv virtualenv 3.10.x thermoswitch-env
pyenv local thermoswitch-env

# 3. Install the package and dependencies via the Makefile
make install

# 4. Run the data extraction pipeline
python -m src.data
```