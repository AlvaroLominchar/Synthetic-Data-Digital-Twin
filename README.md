# Synthetic Data & Digital Twin for Predictive Maintenance

This repository contains the code developed for my Master’s Thesis, focused on synthetic data generation, predictive modeling, and the implementation of a Digital Twin with Streamlit. This approach aims to measure the benefits obtained by leveraging synthetic data along with real data in the design of models for predictive maintenance. By integrating these predictive models into a Digital Twin, it is possible to create a virtual representation of the physical system that not only mirrors its behavior but also anticipates potential failures, enhancing decision-making in predictive maintenance.

---

## 1. Dataset

The dataset used is the AI4I 2020 Predictive Maintenance Dataset.  

- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)  
- Place the file `ai4i2020.xlsx` inside the `data/` folder (this folder is ignored in the repository due to size/licensing reasons).

---

## 2. Installation

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

## 3. Methodological Workflow

The methodology is structured in three main stages:

1. **Individual execution**  
   - Synthetic data generation with each of the following scripts:  
     - `smoteDataGeneration.py`  
     - `gaussianDataGeneration.py`  
     - `ctganDataGeneration.py`  
   - Predictive model training and evaluation using:  
     - `modelTesting.py`  
   - This produces results based on a specific data partition and save them in `executions/individual/<technique>/`.

2. **Generalization execution**  
   - Run `generalizationFlow.py`.  
   - Automates the process of generating synthetic data and evaluating models across multiple partitions. The number of partitions can be specified in TARGET_RUNS. 
   - Provides metrics on robustness and generalization and saves them in `executions/summary/` (metrics and confusion matrices) and `executions/run_xxx/` (each execution's datasets and outputs).

3. **Visualization and interaction**  
   - `exploratoryAnalysis.py` allows exploratory analysis of the dataset, including feature distributions and relationships.
   - `digitalTwin.py` (Streamlit app) provides an interactive interface for exploring the Digital Twin.  

---

## 4. Results

Due to their size (200+ executions, plots, metrics...), the results are not included directly in the repository.  

They can be accessed in this Google Drive folder:  
👉 [Results on Drive](https://drive.google.com/drive/u/0/folders/1cLPekHNyrpkIqrFSZt2z_ZGDAbkEEjcs)

---

## 5. License

Academic use only.  
This code was developed as part of the Master’s Thesis in Business Analytics & Big Data (UAH).
