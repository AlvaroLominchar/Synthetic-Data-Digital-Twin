# Synthetic Data & Digital Twin for Predictive Maintenance

This repository contains the code developed for my Master’s Thesis, focused on **synthetic data generation**, **predictive modeling**, and the implementation of a **Digital Twin** with Streamlit.

---

## 📂 Project Structure

TFM/
├─ dataGeneration/ # Synthetic data generation scripts
│ ├─ smoteDataGeneration.py
│ ├─ gaussianDataGeneration.py
│ ├─ ctganDataGeneration.py
│ └─ utilsDataGeneration.py
│
├─ predictiveModels/ # Predictive models and evaluation
│ ├─ modelTesting.py
│ └─ generalizationFlow.py
│
├─ digitalTwin/ # Digital Twin (Streamlit app)
│ ├─ digitalTwin.py
│ └─ customStreamLit/
│ ├─ bannerUAH.png
│ └─ logoUAH.png
│
├─ exploratoryAnalysis/ # Exploratory data analysis
│ └─ exploratoryAnalysis.py
│
├─ data/ # Dataset (ignored in the repo)
│ └─ ai4i2020.xlsx
│
├─ .gitignore
├─ requirements.txt
└─ README.md


---

## 📊 Dataset

The dataset used is the **AI4I 2020 Predictive Maintenance Dataset**.  
- Source: [UCI Machine Learning Repository](https://doi.org/10.24432/C5HS5C)  
- Place the file `ai4i2020.xlsx` inside the `data/` folder (this folder is ignored in the repository due to size/licensing reasons).

---

## ⚙️ Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/AlvaroLominchar/Synthetic-Data-Digital-Twin.git
cd Synthetic-Data-Digital-Twin
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

--- 

## ▶️ Usage

Examples to run the different modules:

### Exploratory analysis
python exploratoryAnalysis/exploratoryAnalysis.py

### Synthetic data generation
python dataGeneration/smoteDataGeneration.py
python dataGeneration/gaussianDataGeneration.py
python dataGeneration/ctganDataGeneration.py

### Predictive models
python predictiveModels/modelTesting.py
python predictiveModels/generalizationFlow.py

### Digital Twin (Streamlit)
streamlit run digitalTwin/digitalTwin.py

---

## 📂 Results

Due to their size (200+ executions, plots, metrics...), the results are not included directly in the repository.
They can be accessed in this Google Drive folder:
👉 https://drive.google.com/drive/u/0/folders/1cLPekHNyrpkIqrFSZt2z_ZGDAbkEEjcs

---

## 📜 License

Academic use only. This code was developed as part of the Master’s Thesis in Business Analytics & Big Data (UAH).