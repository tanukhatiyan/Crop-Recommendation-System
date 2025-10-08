import streamlit as st
import numpy as np
import pickle
import pandas as pd
import requests
import matplotlib.pyplot as plt

# --------------------- Load trained assets ---------------------
model = pickle.load(open("crop_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))  # Added label encoder

# --------------------- Streamlit page setup ---------------------
st.set_page_config(page_title="🌾 Smart Crop Recommendation System", page_icon="🌱", layout="wide")

st.title("🌾 Smart Crop Recommendation System")
st.write("### Predict the most suitable crop for your soil and weather conditions")

# Tabs
tab1, tab2, tab3 = st.tabs(["🧠 Prediction", "📊 Insights", "🌍 Weather-based Recommendation"])

# ================================================================
# TAB 1: Manual Prediction
# ================================================================
with tab1:
    st.subheader("Enter Soil and Climate Parameters")

    col1, col2, col3 = st.columns(3)
    with col1:
        N = st.number_input("Nitrogen (N)", 0, 150, 50)
        K = st.number_input("Potassium (K)", 0, 210, 50)
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
    with col2:
        P = st.number_input("Phosphorus (P)", 0, 150, 50)
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
        rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0, 100.0)
    with col3:
        temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)

    if st.button("🌱 Predict Crop"):
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        scaled_features = scaler.transform(features)

        encoded_prediction = model.predict(scaled_features)[0]
        predicted_crop = label_encoder.inverse_transform([encoded_prediction])[0]  # Decode to string

        st.success(f"✅ Recommended Crop: **{predicted_crop.capitalize()}**")

        # Crop Info
        crop_info = {
            "rice": "Needs high humidity and rainfall; grows best in clayey or alluvial soil.",
            "maize": "Thrives in warm climate with moderate rainfall and well-drained soil.",
            "wheat": "Requires cool weather and moderate rainfall.",
            "mungbean": "Prefers warm climate; grows well in loamy soil.",
            "apple": "Needs cold climate with well-drained loamy soil.",
            "orange": "Requires subtropical climate; grows in sandy loam soil.",
            "coffee": "Thrives in tropical climate with high rainfall.",
            "mango": "Prefers hot, humid climate; grows in alluvial soil.",
            "grapes": "Needs moderate rainfall and dry climate.",
            "watermelon": "Thrives in warm climate and sandy soil."
        }

        st.info(f"📘 About {predicted_crop.capitalize()}: {crop_info.get(predicted_crop.lower(), 'Information not available.')}")

# ================================================================
# TAB 2: Insights & Visualizations
# ================================================================
with tab2:
    st.subheader("📊 Data Insights & Model Visualization")

    data = pd.read_csv("Crop_recommendation.csv")

    # Crop distribution
    st.write("#### Crop Distribution in Dataset")
    crop_counts = data['label'].value_counts()
    st.bar_chart(crop_counts)

    # Feature importance
    st.write("#### Feature Importance")
    features = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
    importances = model.feature_importances_
    fig, ax = plt.subplots()
    ax.barh(features, importances, color="seagreen")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance in Crop Prediction")
    st.pyplot(fig)

    # Correlation heatmap
    st.write("#### Feature Correlation Heatmap")
    fig2, ax2 = plt.subplots()
    corr = data.corr(numeric_only=True)
    im = ax2.imshow(corr, cmap='YlGn')
    ax2.set_xticks(range(len(corr.columns)))
    ax2.set_yticks(range(len(corr.columns)))
    ax2.set_xticklabels(corr.columns)
    ax2.set_yticklabels(corr.columns)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig2)

# ================================================================
# TAB 3: Weather-based Recommendation
# ================================================================
with tab3:
    st.subheader("🌍 Get Recommendation Based on Live Weather")

    city = st.text_input("Enter City Name (e.g., Lucknow)")
    API_KEY = "7e4f10ac2e533dbfd03762b946b8b4ea"  # Replace with your OpenWeatherMap API key

    if st.button("🌦 Get Weather & Recommend Crop"):
        if city:
            try:
                url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
                response = requests.get(url)
                data = response.json()

                temperature = data["main"]["temp"]
                humidity = data["main"]["humidity"]

                st.info(f"🌡 Temperature: {temperature}°C | 💧 Humidity: {humidity}%")

                # Assumed average soil values
                N, P, K, ph, rainfall = 50, 50, 50, 6.5, 100
                features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
                scaled_features = scaler.transform(features)
                encoded_prediction = model.predict(scaled_features)[0]
                predicted_crop = label_encoder.inverse_transform([encoded_prediction])[0]

                st.success(f"🌾 Based on {city}'s weather, recommended crop: **{predicted_crop.capitalize()}**")

            except Exception as e:
                st.error("⚠️ Error fetching weather data. Please check city name or API key.")
        else:
            st.warning("Please enter a city name.")

# ================================================================
# Footer
# ================================================================
st.markdown("---")
st.caption("Developed by **TANU** | B.Tech Final Year Mini Project 🌱 Data Science Track")
