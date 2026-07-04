import os
import shap
import boto3
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="SpaceX Landing Predictor",
    layout="wide")

@st.cache_resource
def load_model():
    bucket_name = st.secrets["AWS_BUCKET_NAME"] 
    s3_file_path = "production/best_model.pkl"
    local_temp_path = "downloaded_model.pkl"
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            region_name=st.secrets.get("AWS_REGION", "eu-central-1"))
        s3_client.download_file(bucket_name, s3_file_path, local_temp_path)
        return joblib.load(local_temp_path)
    except Exception as e:
        st.error(f"Error connecting to Amazon S3: {e}")
        return None

model = load_model()

st.title("SpaceX Launch Predictor & Cost Optimizer")
st.markdown("Enter the mission parameters in the sidebar to calculate the probability of a successful landing and the expected financial risk.")

st.sidebar.header("Mission Parameters")

launch_site = st.sidebar.selectbox("Launch Site", ["KSC LC 39A", "CCSFS SLC 40", "VAFB SLC 4E"])
booster_version = st.sidebar.selectbox("Booster Version", ["B1058", "B1060", "B1062", "B1067"])
orbit = st.sidebar.selectbox("Orbit", ["LEO", "ISS", "GTO", "VLEO"])

payload_mass_kg = st.sidebar.slider("Payload Mass (kg)", 2000.0, 16000.0, 8000.0)
temp_c = st.sidebar.slider("Temperature (°C)", 0.0, 40.0, 25.0)
wind_speed_kmh = st.sidebar.slider("Wind Speed (km/h)", 0.0, 50.0, 15.0)
visibility_m = st.sidebar.slider("Visibility (m)", 1000, 10000, 10000)

input_df = pd.DataFrame({
    "launch_site": [launch_site],
    "booster_version": [booster_version],
    "orbit": [orbit],
    "payload_mass_kg": [payload_mass_kg],
    "temp_c": [temp_c],
    "wind_speed_kmh": [wind_speed_kmh],
    "visibility_m": [visibility_m]
})

tab1, tab2, tab3 = st.tabs(["Prediction", "Explainability (SHAP)", "Business & ROI"])

if model is None:
    st.error("The file 'best_model.pkl' was not found. Please make sure you have trained and saved your model.")
else:
    prediction_proba = model.predict_proba(input_df)[0][1]
    prediction_class = model.predict(input_df)[0]

    with tab1:
        #Prediction Probability
        st.subheader("Probability Results")
        prob_percentage = prediction_proba * 100        
        col1, col2 = st.columns(2)

        with col1:
            if prediction_class == 1:
                st.success("Likely to Succeed")
            else:
                st.error("Likely to FAIL (High Risk)")
                
        with col2:
            st.metric(label="Probability of Success", value=f"{prob_percentage:.1f}%")


    with tab2:
        #Explainability SHAP (Waterfall Plot)
        st.subheader("Explainability (SHAP)")
        st.markdown("Feature impact on the current prediction (SHAP WATERFALL)")

        try:
            xgb_model = model.named_steps['classifier']
            preprocessor = model.named_steps['preprocess']
            input_processed = preprocessor.transform(input_df)
            feature_names = preprocessor.get_feature_names_out()
            input_shap_df = pd.DataFrame(input_processed, columns=feature_names)
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer(input_shap_df)
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"Failure to create SHAP chart: {e}")

    with tab3:
        #Expected Value (Business Calculator)
        st.subheader("Expected Value (Business Calculator)")
        payload_value = st.number_input("Payload Value (Millions $)", value=150.0)
        launch_cost = st.number_input("Launch Cost (Millions $)", value=67.0)
        failure_cost = st.number_input("Failure Cost (Millions $)", value=50.0)
        prob_success = prediction_proba
        prob_fail = 1.0 - prob_success
        expected_revenue = prob_success * payload_value
        expected_cost = launch_cost + (prob_fail * failure_cost)
        ev = expected_revenue - expected_cost
        st.markdown(f"# Expected Financial Value: **${ev:.2f}M**")
        if ev > 0:
            st.success("The launch is likely to be profitable. Recommended to launch.")
        else:
            st.error("The launch is likely to be unprofitable. The probability of failure is higher than the recommended threshold.")