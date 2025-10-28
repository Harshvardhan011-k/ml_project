import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- Streamlit Page Config ---
st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾", layout="centered")


import os
import joblib
import streamlit as st

# Base directory of the current app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model safely
try:
    model_path = os.path.join(BASE_DIR, 'crop_yield_model.pkl')
    transformer_path = os.path.join(BASE_DIR, 'power_transformer.pkl')
    columns_path = os.path.join(BASE_DIR, 'trained_columns.pkl')

    model = joblib.load(model_path)
    transformer = joblib.load(transformer_path)
    trained_columns = joblib.load(columns_path)
    st.success("✅ Model loaded successfully!")

except FileNotFoundError as e:
    st.error(f"❌ Model file not found: {e}")
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")


# --- Safe Model Loading ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load("crop_yield_model.pkl")
        transformer = joblib.load("power_transformer.pkl")
        trained_columns = joblib.load("trained_columns.pkl")
        return model, transformer, trained_columns, None
    except Exception as e:
        return None, None, None, str(e)

model, transformer, trained_columns, load_error = load_models()

# --- Option Lists ---
states = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
]

crops = [
    "Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Potato", "Tomato",
    "Onion", "Soybean", "Groundnut", "Sunflower", "Chickpea", "Pigeon Pea",
    "Lentil", "Mustard", "Sesame", "Tea", "Coffee", "Rubber", "Coconut"
]

seasons = ["Kharif", "Rabi", "Summer", "Autumn", "Winter", "Whole Year"]

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .hero {
        background-image: url('https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=1600&q=80');
        background-size: cover; background-position: center;
        height: 250px; border-radius: 12px; display: flex;
        flex-direction: column; align-items: center; justify-content: center;
        color: white; text-shadow: 1px 1px 6px rgba(0,0,0,0.6);
        margin-bottom: 30px;
    }
    .hero h1 { font-size: 2.5rem; font-weight: 700; margin: 0; }
    .hero p { font-size: 1.1rem; margin-top: 10px; opacity: 0.95; }
    .prediction-box {
        background-color: white; padding: 2rem 2.5rem;
        border-radius: 15px; box-shadow: 0 0 15px rgba(0,0,0,0.1);
        max-width: 750px; margin: auto;
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #4CAF50, #2E7D32);
        color: white; border: none; border-radius: 10px;
        padding: 0.75rem 2rem; font-size: 1rem; font-weight: 600;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #45a049, #256e29);
        transform: scale(1.03);
    }
    .feature-section {
        display: flex; justify-content: center; gap: 1.5rem; flex-wrap: wrap;
        margin-top: 2rem; margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #ffffff; width: 270px; border-radius: 15px;
        padding: 1.5rem; box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        text-align: center; border: 1px solid #f0f0f0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .feature-card:hover { transform: translateY(-5px); box-shadow: 0 4px 14px rgba(0,0,0,0.12); }
    .feature-icon { font-size: 2rem; margin-bottom: 0.5rem; }
    .feature-title { font-weight: 700; font-size: 1.1rem; color: #333; margin-bottom: 0.3rem; }
    .feature-desc { color: #6b6b6b; font-size: 0.95rem; }
    .footer {
        text-align: center; color: #6b6b6b; font-size: 0.9rem;
        margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #eee;
    }
    </style>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown("""
    <div class="hero">
        <h1>Crop Yield Predictor</h1>
        <p>Harness the power of machine learning to predict agricultural yields with precision</p>
    </div>
""", unsafe_allow_html=True)

# --- Check Model Load ---
if load_error:
    st.error(f"❌ Model could not be loaded: {load_error}")
    st.stop()

# --- Input Form ---
st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
st.subheader("🌿 Crop Yield Prediction")
st.caption("Enter your agricultural data to predict crop yield with machine learning")

col1, col2 = st.columns(2)
with col1:
    state = st.selectbox("🌍 State", ["Select a State"] + states)
    season = st.selectbox("🌦 Season", ["Select a Season"] + seasons)
    pesticide_ml = st.number_input("🧴 Pesticide Used (ml)", min_value=0.0, max_value=10000.0, value=200.0, help="Enter total pesticide quantity used in milliliters.")
with col2:
    crop_year = st.number_input("📅 Crop Year", min_value=2000, max_value=2035, value=2025)
    crop = st.selectbox("🌾 Crop Type", ["Select a Crop"] + crops)
    rainfall = st.number_input("💧 Rainfall (mm)", min_value=0.0, max_value=5000.0, value=1000.0)

st.markdown('</div>', unsafe_allow_html=True)

# --- Prediction Logic ---
st.write("")
if st.button("🚀 Predict Crop Yield"):
    if state == "Select a State" or season == "Select a Season" or crop == "Select a Crop":
        st.warning("⚠️ Please select all dropdown options before predicting.")
    else:
        user_input = {
            "State": state,
            "Crop_Year": float(crop_year),
            "Season": season,
            "Crop": crop,
            "Pesticide": float(pesticide_ml),
            "Annual_Rainfall": float(rainfall)
        }

        input_df = pd.DataFrame([user_input])
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=trained_columns, fill_value=0)

        # Detect multicollinearity warning
        if np.any(input_encoded.corr().abs().values > 0.95):
            st.info("⚠️ High correlation detected between some variables. Model handles this internally.")

        # Predict
        input_transformed = transformer.transform(input_encoded)
        predicted_yield = model.predict(input_transformed)[0]

        st.write("🧠 Encoded Input Preview:")
        st.dataframe(input_encoded)

        # --- Output Section ---
        st.success(f"🌱 **Predicted Crop Yield:** {predicted_yield:.2f} tons/hectare")

        # --- Result Summary ---
        st.markdown("#### 📊 Result Summary")
        st.write(f"""
        - **State:** {state}  
        - **Season:** {season}  
        - **Crop:** {crop}  
        - **Year:** {crop_year}  
        - **Pesticide Used:** {pesticide_ml} ml  
        - **Rainfall:** {rainfall} mm  
        """)

        # --- Visual Insight ---
        st.bar_chart(pd.DataFrame({
            "Parameter": ["Rainfall (mm)", "Pesticide (ml)", "Predicted Yield"],
            "Value": [rainfall, pesticide_ml, predicted_yield]
        }).set_index("Parameter"))

# --- Feature Cards Section ---
st.write("---")
st.markdown("""
    <div class="feature-section">
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Accurate Predictions</div>
            <div class="feature-desc">ML-powered predictions based on historical agricultural data</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Real-time Analysis</div>
            <div class="feature-desc">Get instant yield predictions for better planning</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🌾</div>
            <div class="feature-title">Multiple Crops</div>
            <div class="feature-desc">Support for various crop types and seasons</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">💧</div>
            <div class="feature-title">Sustainability Insights</div>
            <div class="feature-desc">Optimize pesticide use for eco-friendly agriculture</div>
        </div>
    </div>
    <div class="footer">
        Agricultural Yield Prediction System • Powered by Machine Learning
    </div>
""", unsafe_allow_html=True)
