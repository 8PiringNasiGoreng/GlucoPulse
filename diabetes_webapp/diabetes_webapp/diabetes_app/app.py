"""
app.py – Flask Backend untuk Diabetes Prediction
=================================================
Jalankan:  python app.py
Akses:     http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
import joblib, json, os, numpy as np

app = Flask(__name__, static_folder="static", template_folder="templates")

# ── Load Models & Metadata ────────────────────────────────────────────────────
MODELS_DIR = "models"

with open(f"{MODELS_DIR}/metadata.json") as f:
    metadata = json.load(f)

FEATURE_NAMES = metadata["feature_names"]
MODEL_METRICS  = metadata["models"]

scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "KNN"                : "knn.pkl",
    "Decision Tree"      : "decision_tree.pkl",
    "Random Forest"      : "random_forest.pkl",
    "SVM"                : "svm.pkl",
}

models = {}
for name, fname in MODEL_FILES.items():
    path = os.path.join(MODELS_DIR, fname)
    if os.path.exists(path):
        models[name] = joblib.load(path)
        print(f"✅ Loaded: {name}")
    else:
        print(f"⚠️  Not found: {path}")

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

@app.route("/api/models", methods=["GET"])
def get_models():
    """Kembalikan list model + metriknya"""
    result = []
    for name, m in MODEL_METRICS.items():
        result.append({
            "name"      : name,
            "accuracy"  : m["accuracy"],
            "precision" : m["precision"],
            "recall"    : m["recall"],
            "f1"        : m["f1"],
            "roc_auc"   : m["roc_auc"],
            "threshold" : m["threshold"],
        })
    return jsonify(result)

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Body JSON:
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
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    model_name = data.get("model", "Random Forest")
    features   = data.get("features", {})

    if model_name not in models:
        return jsonify({"error": f"Model '{model_name}' tidak tersedia"}), 400

    # Build feature vector sesuai urutan training
    try:
        x = np.array([[features[f] for f in FEATURE_NAMES]], dtype=float)
    except KeyError as e:
        return jsonify({"error": f"Feature tidak lengkap: {e}"}), 400

    # Scale
    x_sc = scaler.transform(x)

    # Predict
    model  = models[model_name]
    thresh = MODEL_METRICS[model_name]["threshold"]
    prob   = model.predict_proba(x_sc)[0][1]
    pred   = int(prob >= thresh)

    return jsonify({
        "model"      : model_name,
        "prediction" : pred,
        "label"      : "Diabetes" if pred == 1 else "Non-Diabetes",
        "probability": round(float(prob), 4),
        "threshold"  : thresh,
        "confidence" : round(float(prob) * 100, 2) if pred == 1
                       else round((1 - float(prob)) * 100, 2),
    })

@app.route("/api/features", methods=["GET"])
def get_features():
    feature_info = [
        {"name": "Pregnancies",             "label": "Jumlah Kehamilan",          "validation_min": 0,    "validation_max": 20,  "step": 1,    "unit": "kali"},
        {"name": "Glucose",                  "label": "Kadar Glukosa",             "validation_min": 50,   "validation_max": 250, "step": 1,    "unit": "mg/dL"},
        {"name": "BloodPressure",            "label": "Tekanan Darah (Diastolik)", "validation_min": 40,   "validation_max": 130, "step": 1,    "unit": "mmHg"},
        {"name": "SkinThickness",            "label": "Ketebalan Kulit",           "validation_min": 5,    "validation_max": 100, "step": 1,    "unit": "mm"},
        {"name": "Insulin",                  "label": "Kadar Insulin",             "validation_min": 10,   "validation_max": 900, "step": 1,    "unit": "mu U/ml"},
        {"name": "BMI",                      "label": "BMI",                       "validation_min": 10,   "validation_max": 70,  "step": 0.1,  "unit": "kg/m²"},
        {"name": "DiabetesPedigreeFunction", "label": "Diabetes Pedigree Function","validation_min": 0.05, "validation_max": 2.5, "step": 0.001,"unit": ""},
        {"name": "Age",                      "label": "Usia",                      "validation_min": 21,   "validation_max": 90,  "step": 1,    "unit": "tahun"},
    ]
    return jsonify(feature_info)

if __name__ == "__main__":
    print("🚀 Diabetes Prediction API running at http://localhost:5000")
    app.run(debug=True, port=5000)
