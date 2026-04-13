import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from PIL import Image

# Streamlit Page Config
st.set_page_config(
    page_title="Ovarian Cancer Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except:
        pass

local_css("style.css")

# --- MODEL LOADING CACHING ---
@st.cache_resource
def load_rf_model():
    rf_model_path = "ovarian_model_top10.pkl"
    if os.path.exists(rf_model_path):
        return joblib.load(rf_model_path)
    return None

@st.cache_resource
def load_dl_model():
    dl_model_path = "ovarian_cancer_densenet.keras"
    if os.path.exists(dl_model_path):
        try:
            return tf.keras.models.load_model(dl_model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return None

st.title("🧬 Ovarian Cancer Prediction Engine")
st.markdown("""
<div class="subtitle-text">
Empowering medical professionals with AI-driven diagnostics.
This system uses two distinct models: a Machine Learning Biomarker model for tabular patient data, and a Deep Learning Histopathology model for tissue images.
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["🩸 Biomarker Analysis", "🔬 Histopathology Image Analysis"])

# --- TAB 1: BIOMARKER MODEL ---
with tab1:
    st.header("Patient Biomarker Entry")
    st.markdown("Enter the top 10 most critical biomarker values to predict the likelihood of Ovarian Cancer.")
    
    rf_model = load_rf_model()
    if rf_model is not None:
        with st.form("biomarker_form"):
            col1, col2 = st.columns(2)
            with col1:
                HE4 = st.number_input("HE4 (pmol/L)", min_value=0.0, max_value=1000.0, value=50.0, step=1.0)
                CA125 = st.number_input("CA125 (U/mL)", min_value=0.0, max_value=5000.0, value=30.0, step=1.0)
                LYM_percent = st.number_input("LYM% (%)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
                Age = st.number_input("Age (Years)", min_value=0, max_value=120, value=45, step=1)
                LYM_count = st.number_input("LYM# (10^9/L)", min_value=0.0, max_value=50.0, value=1.5, step=0.1)
            
            with col2:
                AST = st.number_input("AST (U/L)", min_value=0.0, max_value=500.0, value=20.0, step=1.0)
                NEU = st.number_input("NEU (10^9/L)", min_value=0.0, max_value=50.0, value=4.0, step=0.1)
                CA19_9 = st.number_input("CA19-9 (U/mL)", min_value=0.0, max_value=10000.0, value=15.0, step=1.0)
                CL = st.number_input("CL (mmol/L)", min_value=0.0, max_value=200.0, value=100.0, step=1.0)
                ALP = st.number_input("ALP (U/L)", min_value=0.0, max_value=1000.0, value=70.0, step=1.0)
                
            submit_button = st.form_submit_button(label="Predict Biomarker Outcome")
            
        if submit_button:
            features = pd.DataFrame([[HE4, CA125, LYM_percent, Age, LYM_count, AST, NEU, CA19_9, CL, ALP]],
                                    columns=['HE4', 'CA125', 'LYM%', 'Age', 'LYM#', 'AST', 'NEU', 'CA19-9', 'CL', 'ALP'])
            prediction = rf_model.predict(features)[0]
            
            st.markdown("### Prediction Result")
            if prediction == 1:
                st.error("🚨 HIGH RISK of Ovarian Cancer detected based on biomarkers.")
            else:
                st.success("✅ LOW RISK of Ovarian Cancer detected based on biomarkers.")
                
    else:
        st.warning("Model file 'ovarian_model_top10.pkl' not found! The app requires this model to make predictions. Please ensure it is generated in the workspace.")


# --- TAB 2: HISTOPATHOLOGY MODEL ---
with tab2:
    st.header("Histopathology Image Classifier")
    st.markdown("Upload a histological tissue image to determine if it exhibits tumour characteristics using our DenseNet121 Deep Learning model.")
    
    dl_model = load_dl_model()
    if dl_model is not None:
        uploaded_file = st.file_uploader("Choose a histological image...", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            if st.button("Predict Image"):
                with st.spinner("Analyzing tissue patterns..."):
                    try:
                        # Preprocess image
                        img_resized = image.resize((224, 224))
                        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                        img_array = img_array / 255.0
                        img_array = np.expand_dims(img_array, axis=0) # Create batch axis
                        
                        # Predict
                        pred = dl_model.predict(img_array)[0][0]
                        pred = float(pred) # Convert numpy float to standard float
                        
                        st.markdown("### Diagnosis Result")
                        if pred > 0.5:
                            st.error(f"🚨 TUMOUR DETECTED (Confidence: {pred:.1%})")
                        else:
                            st.success(f"✅ NON-TUMOUR DETECTED (Confidence: {(1 - pred):.1%})")
                    except Exception as e:
                        st.error(f"An error occurred during prediction: {str(e)}")
    else:
        st.info("ℹ️ The Deep Learning model weights (`ovarian_cancer_densenet.keras`) are missing from the current directory. Please upload the generated weights from Kaggle to use this feature.")
