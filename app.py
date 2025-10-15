import streamlit as st
import pickle
import pandas as pd

# --- Page Setup ---
st.set_page_config(page_title="Flood Risk Prediction", page_icon=":umbrella:", layout="wide")
st.title("🌊 Flood Risk Prediction (India)")
st.markdown("Predict flood risk using Logistic Regression or Random Forest.")

# --- Sidebar ---
with st.sidebar:
    st.header("ℹ️ Instructions")
    st.write("Input environmental and local data, then select model and predict.")
    st.write("App by: Daivya, Bramha, Hitesh, Bhavya")

import streamlit as st
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# --- Page Setup ---
st.set_page_config(page_title="Flood Risk Prediction", page_icon=":umbrella:", layout="wide")

_CSS = """
<style>
/* Background gradient */
body {background: linear-gradient(135deg, #0f172a 0%, #0f3666 50%, #0a3d62 100%);} 
.stApp {
  color-scheme: light;
}
/* Card style */
.card {
  background: rgba(255,255,255,0.03);
  border-radius: 14px;
  padding: 18px;
  box-shadow: 0 6px 18px rgba(2,6,23,0.6);
  border: 1px solid rgba(255,255,255,0.04);
}
.big-number {font-size:32px; font-weight:700; color: #e6f7ff}
.muted {color: #cbd5e1}
.success {background: linear-gradient(90deg,#0f766e,#10b981); padding: 12px; border-radius: 10px; color: white}
.danger {background: linear-gradient(90deg,#dc2626,#f97316); padding: 12px; border-radius: 10px; color: white}
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)

header_col1, header_col2 = st.columns([3,1])
with header_col1:
    st.metric(label="Models", value="LogReg + RF")

st.markdown("---")

# --- Sidebar ---
with st.sidebar:
    model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest"]) 
    show_probs = st.checkbox("Show prediction probability (if available)", value=True)
    st.markdown("---")
    preset = st.selectbox("Presets", ["Custom", "Monsoon High Rainfall", "Dry Season", "Urban Low Elevation"])

# --- Input Form ---
def apply_preset(name):
    presets = {
        'Monsoon High Rainfall': {'rainfall':300, 'temperature':28, 'humidity':90, 'river_discharge':1500, 'water_level':6, 'elevation':20, 'population_density':1200, 'infrastructure':0, 'historical_floods':1},
    'Dry Season': {'rainfall':0, 'temperature':35, 'humidity':10, 'river_discharge':20, 'water_level':0.6, 'elevation':500, 'population_density':4000, 'infrastructure':1, 'historical_floods':0},
        'Urban Low Elevation': {'rainfall':120, 'temperature':30, 'humidity':70, 'river_discharge':200, 'water_level':2.5, 'elevation':10, 'population_density':5000, 'infrastructure':0, 'historical_floods':1}
    }
    return presets.get(name, {})

preset_vals = apply_preset(preset)

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    with st.form("flood_form"):
        cols = st.columns([1,1,1,1])
        with cols[0]:
            # Ensure numeric types are consistent: use float for defaults when min_value/step are floats
            rainfall = st.number_input('Rainfall (mm)', value=float(preset_vals.get('rainfall', 50.0)), min_value=0.0, step=1.0)
            humidity = st.slider('Humidity (%)', min_value=0, max_value=100, value=int(preset_vals.get('humidity', 60)))
            river_discharge = st.number_input('River Discharge (m³/s)', value=float(preset_vals.get('river_discharge', 50)), min_value=0.0)
        with cols[1]:
            temperature = st.number_input('Temperature (°C)', value=float(preset_vals.get('temperature', 25)), min_value=-20.0, max_value=60.0, step=0.5)
            water_level = st.number_input('Water Level (m)', value=float(preset_vals.get('water_level', 0.5)), min_value=0.0)
            elevation = st.number_input('Elevation (m)', value=float(preset_vals.get('elevation', 100)), min_value=-1000.0)
        with cols[2]:
            population_density = st.number_input('Population Density (per km²)', value=int(preset_vals.get('population_density', 200)), min_value=0)
            infrastructure = st.selectbox('Infrastructure (0=Poor, 1=Good)', options=[0,1], index=0 if preset_vals.get('infrastructure',0)==0 else 1)
            historical_floods = st.selectbox('Historical Floods (0=No,1=Yes)', options=[0,1], index=preset_vals.get('historical_floods',0))
        with cols[3]:
            st.markdown("### Advanced")
            st.caption("Use advanced sliders to explore sensitivity")
            # quick sensitivity sliders
            rain_scale = st.slider('Rainfall scale', 0.5, 2.0, 1.0, 0.1)
            discharge_scale = st.slider('Discharge scale', 0.5, 3.0, 1.0, 0.1)

        submitted = st.form_submit_button("🔍 Predict Flood Risk")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Safe model loading ---
root = Path(__file__).parent
lr_model = rf_model = scaler = None
lr_path = root / 'lr_classifier_flood.pkl'
rf_path = root / 'rf_classifier_flood.pkl'
scaler_path = root / 'scaler_flood.pkl'

def try_load(p: Path):
    if p.exists():
        try:
            with open(p, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.warning(f'Failed to load {p.name}: {e}')
            return None
    return None

lr_model = try_load(lr_path)
rf_model = try_load(rf_path)
scaler = try_load(scaler_path)

# --- Prediction logic ---
def predict(input_df: pd.DataFrame, model_name: str):
    # Logistic Regression branch: requires scaler
    if model_name == 'Logistic Regression':
        if lr_model is None or scaler is None:
            st.error('Logistic Regression model or scaler not available.')
            return None
        X = scaler.transform(input_df)
        prob = None
        try:
            prob = lr_model.predict_proba(X)[0,1]
        except Exception:
            pass
        pred = lr_model.predict(X)[0]
        return pred, prob
    else:
        if rf_model is None:
            st.error('Random Forest model not available.')
            return None
        prob = None
        try:
            prob = rf_model.predict_proba(input_df)[0,1]
        except Exception:
            pass
        pred = rf_model.predict(input_df)[0]
        return pred, prob

if submitted:
    input_df = pd.DataFrame([{
        'Rainfall (mm)': rainfall * rain_scale,
        'Temperature (°C)': temperature,
        'Humidity (%)': humidity,
        'River Discharge (m³/s)': river_discharge * discharge_scale,
        'Water Level (m)': water_level,
        'Elevation (m)': elevation,
        'Population Density': population_density,
        'Infrastructure': infrastructure,
        'Historical Floods': historical_floods
    }])

    result = predict(input_df, model_choice)
    if result is not None:
        pred, prob = result
        col1, col2 = st.columns([2,3])
        with col1:
            if pred == 1:
                st.markdown("<div class='danger'><h2 style='margin:0'>⚠️ Flood Likely</h2></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='success'><h2 style='margin:0'>✅ No Flood Expected</h2></div>", unsafe_allow_html=True)

            if show_probs and prob is not None:
                st.markdown(f"\n**Predicted probability:** {prob:.1%}")
            st.markdown("\n---\n**Notes:** This is a model-based estimate. Follow local authorities for real-time warnings.")

        with col2:
            st.markdown("**Input summary**")
            st.table(input_df.T.rename(columns={0:'value'}))

        # Small chart: radar-like bar for key features
        st.markdown("---")
        st.markdown("**Feature contributions (simple visualization)**")
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6,2.2))
            feat = ['Rainfall','River Discharge','Water Level','Population Density']
            vals = [input_df['Rainfall (mm)'].iloc[0], input_df['River Discharge (m³/s)'].iloc[0], input_df['Water Level (m)'].iloc[0], input_df['Population Density'].iloc[0]]
            vals = np.array(vals).astype(float)
            vals = (vals - vals.min()) / (vals.max()-vals.min()+1e-6)
            ax.barh(feat, vals, color=['#0ea5a4','#0ea5a4','#f97316','#60a5fa'])
            ax.set_xlim(0,1)
            ax.set_xlabel('Normalized (0-1)')
            st.pyplot(fig)
        except Exception as e:
            st.write('Could not render chart:', e)

        st.success('Prediction complete')

    else:
        st.error('Prediction failed due to missing models or errors. Check logs.')
