"""
app.py – Flask Backend untuk Diabetes Prediction
=================================================
Jalankan:  python app.py
Akses:     http://localhost:5000

Mendukung dua dataset:
  - Dataset 1 (PIMA)   → /api/models, /api/features, /api/predict
  - Dataset 2 (Gejala) → /api/models2, /api/features2, /api/predict2
"""

from flask import Flask, request, jsonify, send_from_directory
import joblib, json, os, numpy as np

app = Flask(__name__, static_folder="static", template_folder="templates")

# ── Absolute base dir (supaya bisa dijalankan dari folder manapun) ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════
# ── DATASET 1 — PIMA (models/) ────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════
MODELS_DIR_V1 = os.path.join(BASE_DIR, "models")

with open(f"{MODELS_DIR_V1}/metadata.json") as f:
    metadata_v1 = json.load(f)

FEATURE_NAMES_V1 = metadata_v1["feature_names"]
MODEL_METRICS_V1  = metadata_v1["models"]

scaler_v1 = joblib.load(f"{MODELS_DIR_V1}/scaler.pkl")

MODEL_FILES_V1 = {
    "Logistic Regression": "logistic_regression.pkl",
    "KNN"                : "knn.pkl",
    "Decision Tree"      : "decision_tree.pkl",
    "Random Forest"      : "random_forest.pkl",
    "SVM"                : "svm.pkl",
}

models_v1 = {}
for name, fname in MODEL_FILES_V1.items():
    path = os.path.join(MODELS_DIR_V1, fname)
    if os.path.exists(path):
        models_v1[name] = joblib.load(path)
        print(f"✅ [v1] Loaded: {name}")
    else:
        print(f"⚠️  [v1] Not found: {path}")

# Range validasi V1 — sesuai get_features_v1
FEATURE_RANGES_V1 = {
    "Pregnancies"            : (0,    20),
    "Glucose"                : (50,   250),
    "BloodPressure"          : (40,   130),
    "SkinThickness"          : (5,    100),
    "Insulin"                : (10,   900),
    "BMI"                    : (10,   70),
    "DiabetesPedigreeFunction": (0.05, 2.5),
    "Age"                    : (21,   90),
}


# ══════════════════════════════════════════════════════════════════════
# ── DATASET 2 — Early Symptom (models_2/) ────────────────────────────
# ══════════════════════════════════════════════════════════════════════
MODELS_DIR_V2 = os.path.join(BASE_DIR, "models_2")

with open(f"{MODELS_DIR_V2}/metadata_2.json") as f:
    metadata_v2 = json.load(f)

FEATURE_NAMES_V2 = metadata_v2["feature_names"]
MODEL_METRICS_V2  = metadata_v2["models"]

scaler_v2 = joblib.load(f"{MODELS_DIR_V2}/scaler_2.pkl")

MODEL_FILES_V2 = {
    "logistic_regression_2": "logistic_regression_2.pkl",
    "knn_2"                : "knn_2.pkl",
    "decision_tree_2"      : "decision_tree_2.pkl",
    "random_forest_2"      : "random_forest_2.pkl",
    "svm_2"                : "svm_2.pkl",
}

models_v2 = {}
for name, fname in MODEL_FILES_V2.items():
    path = os.path.join(MODELS_DIR_V2, fname)
    if os.path.exists(path):
        models_v2[name] = joblib.load(path)
        print(f"✅ [v2] Loaded: {name}")
    else:
        print(f"⚠️  [v2] Not found: {path}")

# Range validasi V2 — hanya age yang punya range; field 0/1 divalidasi terpisah
FEATURE_RANGES_V2 = {
    "age": (16, 90),
}
# Field binary (harus 0 atau 1)
BINARY_FIELDS_V2 = {
    "polyuria", "polydipsia", "sudden_weight_loss", "weakness",
    "polyphagia", "genital_thrush", "visual_blurring", "itching",
    "irritability", "delayed_healing", "partial_paresis",
    "muscle_stiffness", "alopecia", "obesity",
}
VALID_GENDER_V2 = {"male", "female", "0", "1"}


# ══════════════════════════════════════════════════════════════════════
# ── HELPER ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def validate_v1(features):
    """
    Validasi fitur Dataset 1.
    Return list of error strings; kosong berarti valid.
    """
    errors = []
    for fname, (lo, hi) in FEATURE_RANGES_V1.items():
        if fname not in features:
            errors.append(f"'{fname}' tidak ada dalam request")
            continue
        try:
            v = float(features[fname])
        except (TypeError, ValueError):
            errors.append(f"'{fname}' harus berupa angka")
            continue
        if v < lo or v > hi:
            errors.append(f"'{fname}' = {v} di luar rentang yang valid ({lo}–{hi})")
    return errors


def validate_v2(features):
    """
    Validasi fitur Dataset 2.
    Return list of error strings; kosong berarti valid.
    """
    errors = []

    # Validasi gender
    gender_raw = features.get("gender", None)
    if gender_raw is None:
        errors.append("'gender' tidak ada dalam request")
    elif str(gender_raw).strip().lower() not in VALID_GENDER_V2:
        errors.append(f"'gender' harus 'Male' atau 'Female', dapat: '{gender_raw}'")

    # Validasi age
    if "age" in features:
        try:
            v = float(features["age"])
            lo, hi = FEATURE_RANGES_V2["age"]
            if v < lo or v > hi:
                errors.append(f"'age' = {v} di luar rentang yang valid ({lo}–{hi})")
        except (TypeError, ValueError):
            errors.append("'age' harus berupa angka")
    else:
        errors.append("'age' tidak ada dalam request")

    # Validasi field binary
    for fname in BINARY_FIELDS_V2:
        if fname not in features:
            errors.append(f"'{fname}' tidak ada dalam request")
            continue
        try:
            v = float(features[fname])
        except (TypeError, ValueError):
            errors.append(f"'{fname}' harus berupa angka (0 atau 1)")
            continue
        if v not in (0, 1):
            errors.append(f"'{fname}' harus 0 atau 1, dapat: {v}")

    return errors


# ══════════════════════════════════════════════════════════════════════
# ── ROUTES ────────────────────════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


# ── V1 ──────────────────────────────────────────────────────────────

@app.route("/api/models", methods=["GET"])
def get_models_v1():
    result = []
    for name, m in MODEL_METRICS_V1.items():
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


@app.route("/api/features", methods=["GET"])
def get_features_v1():
    feature_info = [
        {"name": "Pregnancies",             "label": "Jumlah Kehamilan",           "type": "number", "min":0,    "max":20,  "step":1,     "unit":"kali"},
        {"name": "Glucose",                  "label": "Kadar Glukosa",              "type": "number", "min":50,   "max":250, "step":1,     "unit":"mg/dL"},
        {"name": "BloodPressure",            "label": "Tekanan Darah (Diastolik)",  "type": "number", "min":40,   "max":130, "step":1,     "unit":"mmHg"},
        {"name": "SkinThickness",            "label": "Ketebalan Kulit",            "type": "number", "min":5,    "max":100, "step":1,     "unit":"mm"},
        {"name": "Insulin",                  "label": "Kadar Insulin",              "type": "number", "min":10,   "max":900, "step":1,     "unit":"mu U/ml"},
        {"name": "BMI",                      "label": "BMI",                        "type": "number", "min":10,   "max":70,  "step":0.1,   "unit":"kg/m²"},
        {"name": "DiabetesPedigreeFunction", "label": "Diabetes Pedigree Function", "type": "number", "min":0.05, "max":2.5, "step":0.001, "unit":""},
        {"name": "Age",                      "label": "Usia",                       "type": "number", "min":21,   "max":90,  "step":1,     "unit":"tahun"},
    ]
    return jsonify(feature_info)


@app.route("/api/predict", methods=["POST"])
def predict_v1():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    model_name = data.get("model", "Random Forest")
    features   = data.get("features", {})

    if model_name not in models_v1:
        return jsonify({"error": f"Model '{model_name}' tidak tersedia"}), 400

    # ── Server-side validation ──
    errors = validate_v1(features)
    if errors:
        return jsonify({"error": "Input tidak valid: " + "; ".join(errors)}), 422

    try:
        x = np.array([[features[f] for f in FEATURE_NAMES_V1]], dtype=float)
    except KeyError as e:
        return jsonify({"error": f"Feature tidak lengkap: {e}"}), 400

    x_sc   = scaler_v1.transform(x)
    model  = models_v1[model_name]
    thresh = MODEL_METRICS_V1[model_name]["threshold"]

    try:
        prob = model.predict_proba(x_sc)[0][1]
    except Exception as e:
        return jsonify({"error": f"Gagal menghitung probabilitas: {str(e)}"}), 500

    pred = int(prob >= thresh)

    return jsonify({
        "model"      : model_name,
        "prediction" : pred,
        "label"      : "Diabetes" if pred == 1 else "Non-Diabetes",
        "probability": round(float(prob), 4),
        "threshold"  : thresh,
        "confidence" : round(float(prob) * 100, 2) if pred == 1
                       else round((1 - float(prob)) * 100, 2),
    })


# ── V2 ──────────────────────────────────────────────────────────────

@app.route("/api/models2", methods=["GET"])
def get_models_v2():
    result = []
    for name, m in MODEL_METRICS_V2.items():
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


@app.route("/api/features2", methods=["GET"])
def get_features_v2():
    feature_info = [
        {"name": "age",                "label": "Usia",                                      "type": "number", "min":16, "max":90, "step":1,   "unit":"tahun"},
        {"name": "gender",             "label": "Jenis Kelamin",                             "type": "select", "options": ["Male", "Female"],   "unit":""},
        {"name": "polyuria",           "label": "Polyuria (sering buang air kecil)",         "type": "number", "min":0,  "max":1,  "step":1,   "unit":"0/1"},
        {"name": "polydipsia",         "label": "Polydipsia (sering haus)",                  "type": "number", "min":0,  "max":1,  "step":1,   "unit":"0/1"},
        {"name": "sudden_weight_loss", "label": "Berat Badan Turun Tiba-tiba",               "type": "number", "min":0,  "max":1,  "step":1,   "unit":"0/1"},
        {"name": "weakness",           "label": "Kelemahan Fisik",                           "type": "number", "min":0,  "max":1,  "step":1,   "unit":"0/1"},
        {"name": "polyphagia",         "label": "Polyphagia (sering lapar)",                 "type": "number", "min":0,  "max":1,  "step":1,   "unit":"0/1"},
        {"name": "genital_thrush",     "label": "Genital Thrush (infeksi jamur)",            "type": "number", "min":0,  "max":1,  "step":1,   "unit":"0/1"},
        {"name": "visual_blurring",    "label": "Penglihatan Kabur",                         "type": "number", "min":0,  "max":1,  "step":1,   "unit":"0/1"},
        {"name": "itching",            "label": "Gatal-gatal",                               "type": "number", "min":0,  "max":1,  "step":1,   "unit":"0/1"},
        {"name": "irritability",       "label": "Mudah Marah / Iritabilitas",                "type": "number", "min":0,  "max":1,  "step":1,   "unit":"0/1"},
        {"name": "delayed_healing",    "label": "Penyembuhan Luka Lambat",                   "type": "number", "min":0,  "max":1,  "step":1,   "unit":"0/1"},
        {"name": "partial_paresis",    "label": "Partial Paresis (lemas sebagian)",          "type": "number", "min":0,  "max":1,  "step":1,   "unit":"0/1"},
        {"name": "muscle_stiffness",   "label": "Kaku Otot",                                 "type": "number", "min":0,  "max":1,  "step":1,   "unit":"0/1"},
        {"name": "alopecia",           "label": "Alopecia (rambut rontok)",                  "type": "number", "min":0,  "max":1,  "step":1,   "unit":"0/1"},
        {"name": "obesity",            "label": "Obesitas",                                   "type": "number", "min":0,  "max":1,  "step":1,   "unit":"0/1"},
    ]
    return jsonify(feature_info)


@app.route("/api/predict2", methods=["POST"])
def predict_v2():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    model_name = data.get("model", "random_forest_2")
    features   = data.get("features", {})

    if model_name not in models_v2:
        return jsonify({"error": f"Model '{model_name}' tidak tersedia"}), 400

    # ── Server-side validation ──
    errors = validate_v2(features)
    if errors:
        return jsonify({"error": "Input tidak valid: " + "; ".join(errors)}), 422

    try:
        feat_vec = []
        for f in FEATURE_NAMES_V2:
            val = features[f]
            if f == "gender":
                val = 1 if str(val).strip().lower() in ("male", "1") else 0
            feat_vec.append(float(val))
        x = np.array([feat_vec], dtype=float)
    except KeyError as e:
        return jsonify({"error": f"Feature tidak lengkap: {e}"}), 400

    x_sc   = scaler_v2.transform(x)
    model  = models_v2[model_name]
    thresh = MODEL_METRICS_V2[model_name]["threshold"]

    try:
        prob = model.predict_proba(x_sc)[0][1]
    except Exception as e:
        return jsonify({"error": f"Gagal menghitung probabilitas: {str(e)}"}), 500

    pred = int(prob >= thresh)

    return jsonify({
        "model"      : model_name,
        "prediction" : pred,
        "label"      : "Diabetes" if pred == 1 else "Non-Diabetes",
        "probability": round(float(prob), 4),
        "threshold"  : thresh,
        "confidence" : round(float(prob) * 100, 2) if pred == 1
                       else round((1 - float(prob)) * 100, 2),
    })


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🚀 Diabetes Prediction API (v1 + v2) running at http://localhost:5000")
    app.run(debug=True, port=5000)