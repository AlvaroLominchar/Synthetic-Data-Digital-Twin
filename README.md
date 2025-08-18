# Synthetic Data & Digital Twin for Predictive Maintenance

This repository contains the code developed for my Master’s Thesis, focused on synthetic data generation, predictive modeling, and the implementation of a Digital Twin with Streamlit.

---

## 📊 Dataset

The dataset used is the AI4I 2020 Predictive Maintenance Dataset.  

- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)  
- Place the file `ai4i2020.xlsx` inside the `data/` folder (this folder is ignored in the repository due to size/licensing reasons).

---

## ⚙️ Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/AlvaroLominchar/Synthetic-Data-Digital-Twin.git
cd Synthetic-Data-Digital-Twin
python -m venv venv
source venv/bin/activate
# On Windows use:
# venv\Scripts\activate
pip install -r requirements.txt
```
---

## 🔄 Methodological Workflow

The methodology is structured in three main stages:

1. **Individual execution**  
   - Synthetic data generation with one of the following scripts:  
     - `smoteDataGeneration.py`  
     - `gaussianDataGeneration.py`  
     - `ctganDataGeneration.py`  
   - Predictive model training and evaluation using:  
     - `modelTesting.py`  
   - This produces results based on a specific data partition.

2. **Generalization execution**  
   - Run `generalizationFlow.py`.  
   - Automates the process of generating synthetic data and evaluating models across multiple partitions.  
   - Provides metrics on robustness and generalization.

3. **Visualization and interaction**  
   - `digitalTwin.py` (Streamlit app) provides an interactive interface for exploring the Digital Twin.  
   - `exploratoryAnalysis.py` allows exploratory analysis of the dataset, including feature distributions and relationships.

---

## 📂 Results

Due to their size (200+ executions, plots, metrics...), the results are not included directly in the repository.  

They can be accessed in this Google Drive folder:  
👉 [Complete results on Drive](https://drive.google.com/drive/u/0/folders/1cLPekHNyrpkIqrFSZt2z_ZGDAbkEEjcs)

---

## 📜 License

Academic use only.  
This code was developed as part of the Master’s Thesis in Business Analytics & Big Data (UAH).
