import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- Page setup for better UX ---
st.set_page_config(
    page_title="Flood Risk Prediction (India)",
    page_icon=":umbrella:",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌊 Flood Risk Prediction For India")
st.markdown("Use this tool to assess flood risk based on weather, geography, and local conditions.")

# --- Sidebar for app info and quick help ---
with st.sidebar:
    st.header("ℹ️ App Instructions")
    st.write("Fill in all fields as per current/local observations or forecasts. Select appropriate categories for land cover and soil type. Click 'Predict Flood Risk' for results.")
    st.markdown("**Model:** Random Forest, trained on Indian regional historical data.")
    st.write("App by: Daivya, Bramha, Hitesh and Bhavya")

# --- Main input form with grouped columns ---
with st.form("flood_form"):
    st.subheader("Input Environmental & Local Data")

    # Split into two columns for cleaner layout
    left, right = st.columns(2)

    with left:
        rainfall = st.number_input('Rainfall (mm)', min_value=0, help="Daily rainfall in mm")
        temperature = st.number_input('Temperature (°C)', min_value=-10, max_value=60, help="Air temperature")
        humidity = st.number_input('Humidity (%)', min_value=0, max_value=100, help="Relative humidity")
        river_discharge = st.number_input('River Discharge (m³/s)', min_value=0, help="Main river discharge rate")
        water_level = st.number_input('Water Level (m)', min_value=0, help="River/water table height")

    with right:
        elevation = st.number_input('Elevation (m)', min_value=-50, help="Above sea level")
        population_density = st.number_input('Population Density', min_value=0, help="People per sq km")
        infrastructure = st.selectbox('Infrastructure', [0, 1], help="1 for present protective infrastructure, 0 for absent")
        historical_floods = st.selectbox('Historical Floods', [0, 1], help="1 if floods often occur in area")
        
        land_covers = ['Agricultural', 'Desert', 'Forest', 'Urban', 'Water Body']
        selected_land_cover = st.selectbox('Land Cover', land_covers, help="Main type surrounding location")
        soil_types = ['Clay', 'Loam', 'Peat', 'Sandy', 'Silt']
        selected_soil_type = st.selectbox('Soil Type', soil_types, help="Main soil composition")

    # One-hot for land cover/soil type
    land_cover_dict = {f'Land Cover_{lc}': 1 if lc == selected_land_cover else 0 for lc in land_covers}
    soil_type_dict = {f'Soil Type_{st}': 1 if st == selected_soil_type else 0 for st in soil_types}

    # Collate all features
    features = {
        'Rainfall (mm)': rainfall,
        'Temperature (°C)': temperature,
        'Humidity (%)': humidity,
        'River Discharge (m³/s)': river_discharge,
        'Water Level (m)': water_level,
        'Elevation (m)': elevation,
        'Population Density': population_density,
        'Infrastructure': infrastructure,
        'Historical Floods': historical_floods,
    }
    features.update(land_cover_dict)
    features.update(soil_type_dict)
    input_df = pd.DataFrame([features])

    submitted = st.form_submit_button("🔍 Predict Flood Risk")

# --- Load model and scaler once, outside form for performance ---
with open('rf_classifier_flood.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler_flood.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# --- Display result ---
if submitted:
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    pred_text = "⚠️ Flood likely! Take precautions." if prediction[0] == 1 else "✅ No flood expected. Stay alert for weather changes."
    st.markdown(f"## Prediction Result\n{pred_text}")

    st.info("This prediction does not account for sudden extreme events. Always follow local advisories.")

