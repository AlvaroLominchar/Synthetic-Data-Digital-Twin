# Import required libraries and functions
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Ignore SDV library warnings
import warnings
warnings.filterwarnings("ignore", message="Trying to unpickle estimator RandomForestClassifier from version 1.5.1 when using version 1.6.1.")
warnings.filterwarnings("ignore", message="Trying to unpickle estimator DecisionTreeClassifier from version 1.5.1 when using version 1.6.1.")
warnings.filterwarnings("ignore", message="Downcasting behavior in `replace` is deprecated and will be removed in a future version.")

# Import the auxiliary utility functions
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from dataGeneration.utilsDataGeneration import load_and_preprocess_dataset
    
# Custom app
APP_TITLE = "Implementación de Gemelo Digital Simple"
TFM_TITLE = "TFM: Técnicas de Exploración de Datos Industriales con IA orientadas a Gemelos Digitales"
AUTHOR = "Álvaro Lominchar González"
AUTHOR2= "Antonio Moratilla Ocaña"
UNIVERSITY = "Universidad de Alcalá"
YEAR = "2025"
ASSETS_DIR = Path(__file__).parent / "customStreamLib"
BANNER_PATH = ASSETS_DIR / "bannerUAH.png"
FAVICON_PATH = ASSETS_DIR / "logoUAH.png"

# Function to track image's paths.
def safe_image(path: Path):
    try:
        if path.exists():
            return str(path)
    except Exception:
        pass
    return None


# Configure dataset path and preprocessing parameters
FILE_PATH = str(Path(__file__).resolve().parent.parent / "data" / "ai4i2020.xlsx")
TARGET = "Machine failure"
DROP_COLS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
CATEGORICAL = ["Type"]
RANDOM_STATE = 42

# Base path for individual execution models
BASE_PATH = ROOT_DIR / "executions" / "individual"

# Load real model
model_real_path = BASE_PATH / "real" / "model" / "model_real.pkl"
if not model_real_path.exists():
    st.error("Modelo real no encontrado. Ejecute primero modelTesting.py de forma individual.")
    st.stop()
model_real = joblib.load(model_real_path)

# Load synthetic models
synthetic_models = {}
for technique in ["smote", "copula", "ctgan"]:
    path = BASE_PATH / technique / "model" / f"model_{technique}.pkl"
    if path.exists():
        synthetic_models[technique.capitalize()] = joblib.load(path)
if not synthetic_models:
    st.error("Modelos sintéticos no encontrados. Ejecute primero modelTesting.py de forma individual.")
    st.stop()

# Load and slpit dataset
X, y, df_clean = load_and_preprocess_dataset(FILE_PATH, TARGET, DROP_COLS, CATEGORICAL)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
test_data = X_test.copy()
test_data[TARGET] = y_test.values

# Set page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=safe_image(FAVICON_PATH) or "🧪",
    layout="wide"
)

# App header (HTML) with TFM information
st.markdown(f"""
<div style="text-align:center; padding-top:4px;">
    <h1 style='margin-bottom:4px; color:#2E86C1;'>{APP_TITLE}</h1>
    <h3 style='margin-top:0; color:#1B4F72;'>{TFM_TITLE}</h3>
    <p style='margin-top:6px; color:#555'>
        <strong>{AUTHOR}</strong> · {UNIVERSITY} · {YEAR}
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Configure sidebar controls (image, parameters and filters)
with st.sidebar:
    banner = safe_image(BANNER_PATH)
    if banner:
        st.image(banner, width=170)  # tamaño pequeño en la barra lateral
    st.header("Configuración")
    num_simulations = st.sidebar.number_input(
        "Nº de puntos a simular",
        min_value=1,
        max_value=len(test_data),
        value=50,
        step=1
    )
    speed = st.sidebar.slider("Segundos de espera por cada punto", 0.5, 3.0, 1.0)
    synthetic_choice = st.sidebar.selectbox("Seleccione el modelo sintético", list(synthetic_models.keys()))

    # Add threshold sliders to the app
    synth_key = f"threshold_synth_{synthetic_choice}"

    # Real model threshold
    st.sidebar.slider(
        "Umbral modelo REAL",
        min_value=0.1, max_value=0.9, step=0.01,
        value=st.session_state.get("threshold_real", 0.5),
        key="threshold_real",
        help="Probabilidad mínima para clasificar como 'Fallo' en el modelo REAL"
    )

    # Synthetic model threshold (updates dynamically with the selected model)
    st.sidebar.slider(
        f"Umbral modelo {synthetic_choice}",
        min_value=0.1, max_value=0.9, step=0.01,
        value=st.session_state.get(synth_key, 0.5),
        key=synth_key,
        help=f"Probabilidad mínima para clasificar como 'Fallo' en el modelo {synthetic_choice}"
    )


# Configure session state
if "index" not in st.session_state:
    st.session_state.index = 0
if "preds_real" not in st.session_state:
    st.session_state.preds_real = []
    st.session_state.preds_synth = []
    st.session_state.true_labels = []
if "target_index" not in st.session_state:
    st.session_state.target_index = num_simulations
if "running" not in st.session_state:
    st.session_state.running = False
if "simulation_complete" not in st.session_state:
    st.session_state.simulation_complete = False

# Define 'Start' amd 'Reset' buttons
if st.sidebar.button("Iniciar"):
    st.session_state.running = True
    st.session_state.target_index = num_simulations

if st.sidebar.button("Reestablecer"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Main execution
if st.session_state.running:

    if st.session_state.index < st.session_state.target_index and st.session_state.index < len(test_data):
        # Set current row based on interation index
        i = st.session_state.index
        row = test_data.iloc[[i]].drop(columns=[TARGET]).reset_index(drop=True)
        true_label = test_data.iloc[i][TARGET]

        # Apply models to define probabilities
        prob_real = model_real.predict_proba(row)[0][1]
        prob_synth = synthetic_models[synthetic_choice].predict_proba(row)[0][1]
        
        # Read specified thresholds for each model
        thr_real = st.session_state.get("threshold_real", 0.5)
        thr_synth = st.session_state.get(f"threshold_synth_{synthetic_choice}", 0.5)
        
        # Make predictions based on threshold
        pred_real = int(prob_real >= thr_real)
        pred_synth = int(prob_synth >= thr_synth)

        # Save to history
        st.session_state.preds_real.append(pred_real)
        st.session_state.preds_synth.append(pred_synth)
        st.session_state.true_labels.append(true_label)
        st.session_state.index += 1

        # Display current data
        placeholder = st.empty()
        with placeholder.container():
            # Data point section
            st.markdown(f"<h3 style='color:#117A65; font-size:22px;'>Instancia #{i+1}</h3>", unsafe_allow_html=True)
            st.dataframe(row)
            st.markdown("<br>", unsafe_allow_html=True)

            # Side-by-side decision diagrams (with probabilities)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div style="border:1px solid #ccc; border-radius:12px; padding:16px; background-color:#fdfdfd;">
                    <div style='text-align:center; font-weight:bold; font-size:16px; margin-bottom:4px;'>Modelo Real – Prob. (umbral {thr_real:.2f})</div>
                    <div style='text-align:center; border:1px solid #ccc; border-radius:10px; padding:10px; width:60px; margin:auto; font-size:22px;'>
                        {prob_real:.2f}
                    </div>
                    <div style='text-align:center; font-size:20px; margin:6px 0;'>↙&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↘</div>
                    <div style="display:flex; justify-content:space-around; gap:15px; margin-top:6px;">
                        <div style="width:150px; text-align:center; padding:6px; border-radius:8px;
                            background-color:{'#27AE60' if pred_real==0 else '#D5D8DC'}; font-size:14px;">
                            ✅ Normal
                        </div>
                        <div style="width:150px; text-align:center; padding:6px; border-radius:8px;
                            background-color:{'#C0392B' if pred_real==1 else '#D5D8DC'}; font-size:14px;">
                            ⚠️ Fallo
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div style="border:1px solid #ccc; border-radius:12px; padding:16px; background-color:#fdfdfd;">
                    <div style='text-align:center; font-weight:bold; font-size:16px; margin-bottom:4px;'>Modelo {synthetic_choice} – Prob. (umbral {thr_synth:.2f})</div>
                    <div style='text-align:center; border:1px solid #ccc; border-radius:10px; padding:10px; width:60px; margin:auto; font-size:22px;'>
                        {prob_synth:.2f}
                    </div>
                    <div style='text-align:center; font-size:20px; margin:6px 0;'>↙&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↘</div>
                    <div style="display:flex; justify-content:space-around; gap:1px; margin-top:6px;">
                        <div style="width:150px; text-align:center; padding:6px; border-radius:8px;
                            background-color:{'#27AE60' if pred_synth==0 else '#D5D8DC'}; font-size:14px;">
                            ✅ Normal
                        </div>
                        <div style="width:150px; text-align:center; padding:6px; border-radius:8px;
                            background-color:{'#C0392B' if pred_synth==1 else '#D5D8DC'}; font-size:14px;">
                            ⚠️ Fallo
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Cumulative metrics table (during simulation)
            real_metrics = {
                "Accuracy": accuracy_score(st.session_state.true_labels, st.session_state.preds_real),
                "F1-Score": f1_score(st.session_state.true_labels, st.session_state.preds_real, zero_division=0),
                "Recall": recall_score(st.session_state.true_labels, st.session_state.preds_real, zero_division=0),
                "Precision": precision_score(st.session_state.true_labels, st.session_state.preds_real, zero_division=0)
            }
            
            synth_metrics = {
                "Accuracy": accuracy_score(st.session_state.true_labels, st.session_state.preds_synth),
                "F1-Score": f1_score(st.session_state.true_labels, st.session_state.preds_synth, zero_division=0),
                "Recall": recall_score(st.session_state.true_labels, st.session_state.preds_synth, zero_division=0),
                "Precision": precision_score(st.session_state.true_labels, st.session_state.preds_synth, zero_division=0)
            }
            
            metrics_df = pd.DataFrame({
                "Real Model": {k: round(v, 2) for k, v in real_metrics.items()},
                f"{synthetic_choice} Model": {k: round(v, 2) for k, v in synth_metrics.items()}
            })
            
            st.markdown("<h3 style='color:#117A65; margin-top:40px; font-size:22px; margin-bottom:20px;'>Métricas actuales</h3>", unsafe_allow_html=True)
            st.dataframe(metrics_df.style.format("{:.2f}"), use_container_width=True)

            # Build charts for a better understanding
            chart_col1, chart_col2 = st.columns(2)
            
            # Heatmap of predictions vs ground truth
            heatmap_data = pd.DataFrame({
                "Etiqueta real": st.session_state.true_labels,
                "Modelo real": st.session_state.preds_real,
                f"Modelo {synthetic_choice}": st.session_state.preds_synth
            })
            
            # Map values to string labels for better readability
            label_map = {0: "Normal", 1: "Failure"}
            heatmap_data_str = heatmap_data.map(lambda x: label_map.get(x, x))
            
            # Encode to numerical for heatmap color
            heatmap_encoded = heatmap_data_str.replace({"Normal": 0, "Failure": 1})
            heatmap_encoded = heatmap_encoded.astype(int)
            
            # Create figure and plot
            fig, ax = plt.subplots(figsize=(10, 2.5))
            sns.heatmap(
                heatmap_encoded.T,
                cmap=["#27AE60", "#C0392B"],
                cbar=False,
                linewidths=0.5,
                linecolor='white',
                xticklabels=False,
                yticklabels=heatmap_encoded.columns,
                ax=ax
            )
            
            ax.set_title("Predicciones vs Realidad (Verde=Normal, Rojo=Fallo)", fontsize=12)
            ax.set_xlabel("Instancia de test")
            ax.set_ylabel("")
            
            st.markdown("<h4 style='color:#117A65; font-size:22px; margin-top:30px;'>Mapa de calor comparativo</h4>", unsafe_allow_html=True)
            st.pyplot(fig)

        # Delay before next data point
        time.sleep(speed)

        # Trigger next render
        st.rerun()

    else:
        st.success("¡Simulación completada!")
        st.session_state.running = False
        st.session_state.simulation_complete = True
        st.rerun()

# Information displayed when the simulation is finished
elif st.session_state.simulation_complete:
    
    # Update final metrics for each model
    real_metrics = {
        "Accuracy": accuracy_score(st.session_state.true_labels, st.session_state.preds_real),
        "F1-Score": f1_score(st.session_state.true_labels, st.session_state.preds_real, zero_division=0),
        "Recall": recall_score(st.session_state.true_labels, st.session_state.preds_real, zero_division=0),
        "Precision": precision_score(st.session_state.true_labels, st.session_state.preds_real, zero_division=0)
    }

    synth_metrics = {
        "Accuracy": accuracy_score(st.session_state.true_labels, st.session_state.preds_synth),
        "F1-Score": f1_score(st.session_state.true_labels, st.session_state.preds_synth, zero_division=0),
        "Recall": recall_score(st.session_state.true_labels, st.session_state.preds_synth, zero_division=0),
        "Precision": precision_score(st.session_state.true_labels, st.session_state.preds_synth, zero_division=0)
    }

    metrics_df = pd.DataFrame({
        "Real Model": {k: round(v, 2) for k, v in real_metrics.items()},
        f"{synthetic_choice} Model": {k: round(v, 2) for k, v in synth_metrics.items()}
    })

    st.markdown("<h2 style='text-align:left; color:#117A65; font-size:22px; margin-top:40px;'>Resumen de métricas finales</h2>", unsafe_allow_html=True)
    st.dataframe(metrics_df.style.format("{:.2f}"), use_container_width=True)

    # Build the list of recommended predictive maintenance
    st.markdown("<h2 style='text-align:left; color:#117A65;font-size:22px;  margin-top:40px;'>Plan de Acción basado en Mantenimiento Predictivo</h2>", unsafe_allow_html=True)

    # Model filter for maintenance list
    model_choice = st.selectbox(
        "Selecciona el modelo para ver los casos de mantenimiento sugerido:",
        options=["Real", synthetic_choice],
        index=0,
        key="recommendation_model_selector"
    )

    recommendation_text = "🔧 Revisión inmediata recomendada del sistema."

    # Custom list for each model
    if model_choice == "Real":
        recommendations = [
            {"Instancia": i+1, "Acción Recomendada": recommendation_text}
            for i, pred in enumerate(st.session_state.preds_real)
            if pred == 1
        ]
    else:
        recommendations = [
            {"Instancia": i+1, "Acción Recomendada": recommendation_text}
            for i, pred in enumerate(st.session_state.preds_synth)
            if pred == 1
        ]

    df_recommend = pd.DataFrame(recommendations)

    if not df_recommend.empty:
        st.dataframe(df_recommend, hide_index=True, use_container_width=True)
    else:
        st.success(f"✅ Ningún fallo detectado por el modelo {model_choice}.")
    
else:
    st.info(
        """El proceso diseñado trata de representar el funcionamiento real de un Gemelo Digital de forma simplificada. Una implementación real y completa de un DT para mantenimiento predictivo debería incluir además cierta modelación física de la máquina en cuestión, además de una generación de datos y toma de decisiones más complejas. El único objetivo del desarrollo actual es facilitar el entendimiento del proceso interno de un DT."""
    )


# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align:center; color:#777; font-size:0.9rem; padding:8px 0 20px 0;">
    {APP_TITLE} — {UNIVERSITY} · {YEAR}<br/>
    Desarrollado por <strong>{AUTHOR}</strong>. Tutorizado por <strong>{AUTHOR2}</strong><br/>
    <span style="font-size:0.85rem;">Demo académica para TFM. No se debe usar sin validación adicional.</span>
</div>
""", unsafe_allow_html=True)