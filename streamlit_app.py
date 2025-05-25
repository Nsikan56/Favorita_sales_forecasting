import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import shap
from streamlit_shap import st_shap

# Set page config
st.set_page_config(page_title="Favorita Sales Forecaster", layout="centered")

# Load trained model
model_path = os.path.join("models", "xgb_favorita_model.pkl")
model = joblib.load(model_path)

st.title("Favorita Sales Forecast🛒")
st.markdown("Predict daily sales for a given store and item using a trained XGBoost model.")

# --- Input Form ---
with st.form("input_form"):
    st.subheader("📥 Input Features")

    store_nbr = st.number_input("Store Number", min_value=1, max_value=54, value=1)
    onpromotion = st.selectbox("Is the item on promotion?", ["Yes", "No"])
    day = st.number_input("Day of Month", min_value=1, max_value=31, value=15)
    month = st.number_input("Month", min_value=1, max_value=12, value=6)
    year = st.number_input("Year", min_value=2017, max_value=2017, value=2017)

    # Submit button inside form
    submitted = st.form_submit_button("🔮 Predict Sales")

# --- Handle Prediction ---
if submitted:
    # Step 1: Build a blank row with all expected columns
    full_input = pd.DataFrame(columns=model.feature_names_in_)
    full_input.loc[0] = 0  # Initialize all features to 0

    # Step 2: Fill values based on form
    full_input.at[0, "store_nbr"] = store_nbr
    full_input.at[0, "onpromotion"] = 1 if onpromotion == "Yes" else 0
    full_input.at[0, "day"] = day
    full_input.at[0, "month"] = month
    full_input.at[0, "year"] = year

    # Step 3: Fill in default values (example values; adjust as needed)
    full_input.at[0, "cluster"] = 13
    full_input.at[0, "dcoilwtico"] = 60.0
    full_input.at[0, "is_holiday"] = 0
    full_input.at[0, "transactions"] = 1500
    full_input.at[0, "lag_1"] = 200
    full_input.at[0, "lag_7"] = 180
    full_input.at[0, "lag_14"] = 190
    full_input.at[0, "rolling_mean_7"] = 185
    full_input.at[0, "rolling_mean_14"] = 183

    # Optional one-hot overrides (set only 1 per category if used)
    # full_input.at[0, "family_BEVERAGES"] = 1
    # full_input.at[0, "city_Quito"] = 1
    # full_input.at[0, "type_D"] = 1

    # Step 4: Predict using the trained model
    prediction = model.predict(full_input)[0]
    
    # Step 5: Display prediction
    st.success(f"📈 Predicted Sales: {prediction:.0f} units")
    # SHAP explanation
    st.subheader("🔍 Why did the model predict that?")

    # Create SHAP explainer and values
    explainer = shap.Explainer(model)
    shap_values = explainer(full_input)

    # Display SHAP waterfall plot for this prediction
    st_shap(shap.plots.waterfall(shap_values[0]), height=400)
    st.markdown("""
        🧠 **Interpretation:**  
        This SHAP plot shows how each feature influenced the model’s prediction.  
        In this case, recent sales patterns (`lag_1`, `lag_7`, `rolling_mean_7`) decreased the forecast,  
        while being on promotion and store effects added a smaller upward influence.

        Together, these explain why the predicted daily sales landed at this value.
        """)


