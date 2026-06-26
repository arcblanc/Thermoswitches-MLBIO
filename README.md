# Thermoswitches-ML


# 🧬 RNA Thermoswitch Engineering: A Machine Learning & Biophysics Pipeline

**Author:** Amier Zuhri 
**Role:** Biotech Machine Learning Engineer  
**Domain:** Synthetic Biology, Bioinformatics, Machine Learning, RNA  

## 📌 Project Overview
This project aims to computationally classify and engineer **prokaryotic RNA thermoswitches** (RNA thermometers). Biologically, these are highly structured *cis*-regulatory non-coding RNA elements (often located in the 5' UTR) that fold into hairpins at low temperatures to physically sequester the Shine-Dalgarno (SD) sequence and prevent ribosome binding. Upon a temperature increase, the hairpin melts in a zipper-like fashion, exposing the SD sequence and activating translation. 

**The Engineering Goal:** Build a robust machine learning pipeline(Random Forest) to classify pure non leaky thermoswitches at a specific temperature, with the ultimate future goal of utilizing deep learning foundation models to engineer a synthetic thermoswitch that triggers precisely at 55°C with zero low-temperature "leakiness" and have a very sharp hill coefficient.

---

## 🏗️ Architecture
This repository is structured as a modular Python pipeline under `src/`:

| Module | Role |
|--------|------|
| `data_extraction.py` | Rfam SQL extraction (positives + negatives) |
| `sequence_retrieval.py` | NCBI Entrez FASTA fetch |
| `cd_hit_sequence_similarity.py` | CD-HIT homology filtering |
| `knn_undersample.py` | k-mer ENN + RUS balancing |
| `vienna_rna.py` | ViennaRNA melting / unpaired-probability features |
| `nupack.py` | NUPACK test-tube ensemble features |
| `thermo_common.py` | Shared dataset loading, temp grid, Hill-fit stubs |

---

## 🛠️ Phase 1: Data Extraction & Balancing Workflow
To prevent the model from learning evolutionary background noise or falling victim to class imbalance, we employ a strict, multi-step data processing pipeline:

1. **Broad SQL Extraction (Rfam):** 
   *   **Positives:** ~2,960 known prokaryotic, heat-shock thermoswitches (e.g., ROSE, FourU elements) extracted from the Rfam relational database.
   *   **Negatives:** A massive pool (~168,000) of standard, non-temperature-responsive bacterial 5' UTRs and *Cis*-regulatory elements from the same biological neighborhood. 
2. **FASTA Retrieval:** We utilize Biopython's `Entrez` module to fetch the raw nucleotide sequences (A, C, G, U) using the genomic coordinates (`seq_start`, `seq_end`) from Rfam.
3. **Homology Filtering (CD-HIT):** *Before* balancing, both positive and negative FASTA datasets are passed independently through the **CD-HIT algorithm**. By setting a sequence identity threshold (e.g., 80%), we remove highly homologous and redundant sequences that would otherwise cause severe data leakage and overfitting.

4. **Undersampling** using **K-mer ENN + RUS** architecture : To resolve the severe class imbalance between the surviving ~1,000 positive thermoswitches and the massive negative majority class, we apply an advanced k-mer ENN + RUS balancing pipeline. Machine learning algorithms trained on heavily skewed data tend to predict the over-represented class poorly, making balancing a critical step. 
   * First, the script translates the raw RNA text strings into numerical feature matrices using k-mer frequency counts. 
   
   * Next, it applies Edited Nearest Neighbors (ENN), a specialized K-Nearest Neighbors (KNN) technique that calculates the mathematical distance between data points in the feature space. 

   * Rather than blindly dropping sequences, ENN strategically evaluates the k-mer features to maximize the distance between the selected negative training data points. This guarantees that the negative non-switches we retain are from completely different corners of the feature space, ensuring they are as structurally diverse and distinct from one another as possible to preserve maximum non-redundant information. 
   
   * Finally, Random Under-Sampling (RUS) shrinks the ENN-cleaned negative pool to match the positive class (strict 1:1 ratio). This outputs the finalized golden dataset to `data/processed/balanced/balanced_dataset.*`, sized for dual thermodynamic feature extraction (ViennaRNA + NUPACK).

---

## 🌡️ Phase 2: Biophysical Feature Engineering (Dual-Engine)
Static nucleotide strings cannot convey temperature dynamics to classical algorithms. We run **two independent thermodynamics engines** on the balanced dataset, each producing its own feature table:

| Track | Tooling | Role |
|-------|---------|------|
| **ViennaRNA** | [`RNAheat`](https://www.tbi.univie.ac.at/RNA/documentation.html), `RNAplfold` (Python `RNA` / `viennarna` bindings) | Native melting curves across a temperature range; locally stable **single-strand / unpaired sub-state probabilities** along the sequence (RBS-exposure proxy) |
| **NUPACK** | `Model`, `Tube`, `tube_analysis` ([NUPACK 4.1 Python API](https://docs.nupack.org/4.1/)) | Test-tube ensemble analysis across 20–70°C; complementary multi-strand thermodynamic view |

**Shared post-processing (both tracks):** Each engine's melting/exposure curve is fitted to a logistic sigmoid (Hill function) to extract:
   *   **Midpoint ($T_m$):** The inflection point determining the activation temperature.
   *   **Hill Coefficient (Slope):** The steepness of the curve, defining how "zipper-like" or digital the switch is.
   *   **Amplitude:** The difference between the top and bottom asymptotes, representing switch strength and low-temperature leakiness.

**Outputs:**
   * `data/processed/viennarna/features.csv` — `viennarna_*` columns
   * `data/processed/nupack/features.csv` — `nupack_*` columns

> **License note:** NUPACK requires an active paid subscription per the [NUPACK 4 license terms](https://docs.nupack.org/4.1/). ViennaRNA is open-source via [bioconda](https://www.tbi.univie.ac.at/RNA/documentation.html) or `pip install viennarna`.

---

## 🤖 Phase 3: Machine Learning Classification
To classify functional switches based on our engineered features, we employ an ensemble learning approach:
*   **The Algorithm:** **Random Forest Classifier**.
*   **Dual-input design:** The classifier receives **two independent feature blocks** — `viennarna_*` and `nupack_*` columns — joined to the balanced golden dataset on `(rfamseq_acc, seq_start, seq_end)`. Each engine's Hill-fit parameters and summary statistics are fed separately so the forest can learn which thermodynamic view is most discriminative.
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

```bash
# 1. Clone the repository
git clone https://github.com/arcblanc/Thermoswitches-MLBIO.git
cd Thermoswitches-MLBIO

# 2. Create conda/micromamba environment (includes cd-hit, viennarna)
conda env create -f environment.yml
conda activate thermoswitches-mlbio

# Or use a venv with pip (cd-hit and viennarna must be installed separately)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure NCBI credentials for sequence retrieval
cp .env.example .env   # set EMAIL and NCBI_API_KEY

# 4. Run the data pipeline
python src/data_extraction.py
python src/sequence_retrieval.py --all
python src/cd_hit_sequence_similarity.py --all
python src/knn_undersample.py

# 5. Thermodynamics feature extraction (architecture scaffold — dry-run by default)
python src/vienna_rna.py --dry-run
python src/nupack.py --dry-run
```

**Optional dependencies for Phase 2:**
* `viennarna` — bioconda or `pip install viennarna` ([ViennaRNA docs](https://www.tbi.univie.ac.at/RNA/documentation.html))
* `nupack` — `pip install nupack` ([NUPACK 4.1 docs](https://docs.nupack.org/4.1/); requires paid subscription)