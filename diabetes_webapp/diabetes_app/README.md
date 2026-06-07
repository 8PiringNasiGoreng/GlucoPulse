# 🩺 Diabetes Prediction Web App

Aplikasi web prediksi diabetes menggunakan 5 model ML dengan Flask backend + HTML frontend.

---

## 📁 Struktur Folder

```
diabetes_app/
├── train_and_save.py     ← Jalankan PERTAMA untuk melatih & save model
├── app.py                ← Flask server (backend)
├── requirements.txt      ← Daftar library
├── Training.csv          ← ⚠️ Taruh di sini (dari dataset kamu)
├── Testing.csv           ← ⚠️ Taruh di sini (dari dataset kamu)
├── models/               ← Folder output model (otomatis dibuat)
│   ├── scaler.pkl
│   ├── logistic_regression.pkl
│   ├── knn.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── svm.pkl
│   └── metadata.json
└── templates/
    └── index.html        ← Frontend web
```

---

## 🚀 Cara Pakai

### STEP 1 — Install Dependencies

Buka terminal/CMD di folder `diabetes_app/`, lalu:

```bash
pip install -r requirements.txt
```

---

### STEP 2 — Siapkan Dataset

Copy file `Training.csv` dan `Testing.csv` ke folder `diabetes_app/`  
(file yang sama dengan yang dipakai di notebook).

---

### STEP 3 — Train & Save Model

Jalankan sekali untuk melatih semua model:

```bash
python train_and_save.py
```

Kalau berhasil, akan muncul output seperti:
```
✅ Logistic Regression | Recall=0.87 | F1=0.78 | Threshold=0.30
✅ KNN | Recall=0.88 | F1=0.76 | ...
...
✅ Semua model tersimpan di folder 'models/'
```

> ⚠️ **HANYA perlu dijalankan SATU KALI.** Setelah model tersimpan,
> kamu tidak perlu train ulang kecuali ada perubahan data/kode.

---

### STEP 4 — Jalankan Web Server

```bash
python app.py
```

Output:
```
🚀 Diabetes Prediction API running at http://localhost:5000
```

---

### STEP 5 — Buka Browser

Buka: **http://localhost:5000**

---

## 🌐 API Endpoints

| Method | URL | Fungsi |
|--------|-----|--------|
| GET | `/` | Halaman web |
| GET | `/api/models` | List semua model + metrik |
| GET | `/api/features` | Info fitur input |
| POST | `/api/predict` | Prediksi (JSON body) |

### Contoh Request Prediksi (POST `/api/predict`):
```json
{
  "model": "Random Forest",
  "features": {
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 25,
    "Insulin": 80,
    "BMI": 28.5,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 35
  }
}
```

---

## 🔧 Troubleshooting

**Error "ModuleNotFoundError"** → Jalankan `pip install -r requirements.txt`

**Error "Training.csv not found"** → Pastikan file CSV ada di folder yang sama dengan `train_and_save.py`

**Error "models/ folder not found"** → Jalankan `train_and_save.py` dulu

**Port 5000 sudah dipakai** → Ubah di `app.py` baris terakhir: `app.run(debug=True, port=5001)`
