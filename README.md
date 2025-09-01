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
source venv/bin/activate # On Windows use: venv\Scripts\activate
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
   - This process generates four Random Forest models along with three synthetic datasets. One model is trained on real data, while the other three are trained on each synthetic dataset. Afterwards, all models are tested on the same subset of real data, corresponding to the initial partition of the dataset, which is synchronized across all scripts. The results are then produced based on this specific data partition and saved in `executions/individual/<technique>/` (default `random_state` for this part is 42).

2. **Generalization execution**  
   - Run `generalizationFlow.py`.  
   - This stage automates the full experimental pipeline — generating synthetic data, training predictive models, and evaluating them — across multiple random partitions of the dataset.  
   - The number of executions is defined by the parameter `TARGET_RUNS`. For each run, synthetic datasets are generated, models are trained, and evaluation metrics are collected.  
   - Once all runs are completed, results are aggregated to compute averages and variability, providing a more robust estimation of model performance and the benefits of synthetic data.  
   - Outputs are stored in two locations:  
     - `executions/summary/`: aggregated metrics and confusion matrices.  
     - `executions/run_xxx/`: datasets, intermediate files, and outputs for each individual execution.  

3. **Visualization and interaction**  
   - `exploratoryAnalysis.py` provides tools for exploring the dataset, including visualizations of feature distributions, correlations, and dimensionality reduction. These analyses help to understand the structure of the real dataset before generating synthetic data.  
   - `digitalTwin.py` is implemented as a Streamlit application that integrates the predictive models into a user-friendly interface. The Digital Twin allows interactive exploration of the system’s behavior, visualizing predictions, monitoring synthetic versus real data performance, and simulating different operational scenarios for predictive maintenance. In this case, you should start the script with `streamlit run digitalTwin.py/` rather than the typical `python3 digitalTwin.py/`, as Streamlit manages the application interface.

---

## 4. Results

Due to their size (200+ executions, plots, metrics...), the results are not included directly in the repository.  

They can be accessed in this Google Drive folder:  
👉 [Results on Drive](https://drive.google.com/drive/u/0/folders/1cLPekHNyrpkIqrFSZt2z_ZGDAbkEEjcs)

---

## 5. License

Academic use only.  
This code was developed as part of the Master’s Thesis in Business Analytics & Big Data (UAH).