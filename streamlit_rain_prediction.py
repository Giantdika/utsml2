
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load model dan preprocessing tools
model = load_model("rain_prediction_model.h5")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Prediksi Hujan Berdasarkan Data Cuaca")

st.markdown("Masukkan data cuaca berikut untuk memprediksi apakah akan terjadi hujan.")

min_temp = st.number_input("Min Temp (°C)", value=10.0)
max_temp = st.number_input("Max Temp (°C)", value=25.0)
humidity_9am = st.number_input("Humidity 9am (%)", value=70.0)
humidity_3pm = st.number_input("Humidity 3pm (%)", value=50.0)
pressure_9am = st.number_input("Pressure 9am (hPa)", value=1012.0)
pressure_3pm = st.number_input("Pressure 3pm (hPa)", value=1008.0)

if st.button("Prediksi"):
    sample = np.array([[min_temp, max_temp, humidity_9am, humidity_3pm, pressure_9am, pressure_3pm]])
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)
    result = le.inverse_transform([int(prediction[0] > 0.5)])
    st.success(f"Prediksi: {'Akan Hujan' if result[0] == 'Yes' else 'Tidak Hujan'}")
