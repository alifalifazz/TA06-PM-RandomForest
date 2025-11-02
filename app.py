from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# === 1. Muat model Random Forest untuk Diabetes ===
try:
    model = joblib.load('model/diabetes_rf_model.joblib')
except Exception as e:
    print(f"Gagal memuat model: {e}")
    model = None

# === 2. Urutan fitur sesuai dataset diabetes.csv ===
feature_names = [
    'Pregnancies', 
    'Glucose', 
    'BloodPressure', 
    'SkinThickness', 
    'Insulin', 
    'BMI', 
    'DiabetesPedigreeFunction', 
    'Age'
]

# === 3. Halaman utama ===
@app.route('/')
def home():
    return render_template('index.html')

# === 4. Proses prediksi ===
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('result.html',
                               prediction="❌ Model belum dimuat dengan benar. Periksa file 'model/diabetes_rf_model.joblib'")

    try:
        # Ambil input dari form sesuai urutan fitur
        input_data = [float(request.form[name]) for name in feature_names]
        final_input = np.array(input_data).reshape(1, -1)

        # Prediksi menggunakan model Random Forest
        pred = model.predict(final_input)[0]

        # Interpretasi hasil prediksi (0 = tidak berisiko, 1 = berisiko diabetes)
        if pred == 1:
            prediction = '⚠️ Pasien Berisiko Diabetes'
        else:
            prediction = '✅ Pasien Tidak Berisiko Diabetes'

    except Exception as e:
        print(f"Error: {e}")
        prediction = "⚠️ Terjadi kesalahan input. Pastikan semua kolom diisi angka dengan benar."

    # Tampilkan hasil prediksi di halaman hasil
    return render_template('result.html', prediction=prediction)

# === 5. Jalankan Flask App ===
if __name__ == '__main__':
    app.run(debug=True)
