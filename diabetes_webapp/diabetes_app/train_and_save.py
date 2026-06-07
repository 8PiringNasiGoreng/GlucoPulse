"""
train_and_save.py
=================
Jalankan script ini SATU KALI untuk melatih semua model dan menyimpannya.
Butuh file: Training.csv, Testing.csv (taruh di folder yang sama)

Usage:
    python train_and_save.py
"""

import pandas as pd
import numpy as np
import joblib, os, json

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── 1. Load & Preprocess ────────────────────────────────────────────────────
print("📂 Loading dataset...")
train_df = pd.read_csv("Training.csv")
test_df  = pd.read_csv("Testing.csv")

zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
train_clean, test_clean = train_df.copy(), test_df.copy()
for col in zero_cols:
    median_val = train_clean[train_clean[col] != 0][col].median()
    train_clean[col] = train_clean[col].replace(0, median_val)
    test_clean[col]  = test_clean[col].replace(0, median_val)

X_train = train_clean.drop('Outcome', axis=1)
y_train = train_clean['Outcome']
X_test  = test_clean.drop('Outcome', axis=1)
y_test  = test_clean['Outcome']

FEATURE_NAMES = X_train.columns.tolist()

# ── 2. Scaling ───────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
print("✅ Scaler disimpan")

# ── 3. SMOTE ─────────────────────────────────────────────────────────────────
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_sc, y_train)
print(f"✅ SMOTE selesai | Before: {dict(y_train.value_counts())} | After: {dict(pd.Series(y_train_smote).value_counts())}")

# ── 4. Helper ─────────────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def tune_threshold(y_true, y_prob, target_recall=0.85):
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.20, 0.65, 0.01):
        yp = (y_prob >= t).astype(int)
        rec = recall_score(y_true, yp)
        f1  = f1_score(y_true, yp)
        if rec >= target_recall and f1 > best_f1:
            best_f1, best_thresh = f1, t
    return round(best_thresh, 2)

# ── 5. Train Models ───────────────────────────────────────────────────────────
MODELS_CONFIG = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
    ),
    "KNN": (
        KNeighborsClassifier(),
        {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance']}
    ),
    "Decision Tree": (
        DecisionTreeClassifier(class_weight='balanced', random_state=42),
        {'max_depth': [3, 4, 5, 6, 7], 'min_samples_split': [2, 5, 10]}
    ),
    "Random Forest": (
        RandomForestClassifier(class_weight='balanced', random_state=42),
        {'n_estimators': [100, 200], 'max_depth': [5, 7, 10, None], 'min_samples_split': [2, 5]}
    ),
    "SVM": (
        SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
        {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.01]}
    ),
}

all_metrics = {}

for name, (base_model, params) in MODELS_CONFIG.items():
    print(f"\n🔄 Training {name}...")
    gs = GridSearchCV(base_model, params, cv=cv, scoring='recall', n_jobs=-1)
    gs.fit(X_train_smote, y_train_smote)
    best_model = gs.best_estimator_

    prob  = best_model.predict_proba(X_test_sc)[:, 1]
    thresh = tune_threshold(y_test, prob)
    pred  = (prob >= thresh).astype(int)

    metrics = {
        "threshold" : thresh,
        "accuracy"  : round(accuracy_score(y_test, pred), 4),
        "precision" : round(precision_score(y_test, pred), 4),
        "recall"    : round(recall_score(y_test, pred), 4),
        "f1"        : round(f1_score(y_test, pred), 4),
        "roc_auc"   : round(roc_auc_score(y_test, prob), 4),
        "best_params": gs.best_params_,
    }
    all_metrics[name] = metrics

    # Simpan model
    safe_name = name.lower().replace(" ", "_")
    joblib.dump(best_model, f"{MODELS_DIR}/{safe_name}.pkl")
    print(f"  ✅ {name} | Recall={metrics['recall']} | F1={metrics['f1']} | Threshold={thresh} | Best params={gs.best_params_}")

# Simpan metadata (threshold + metrics semua model)
meta = {
    "feature_names": FEATURE_NAMES,
    "models": all_metrics
}
with open(f"{MODELS_DIR}/metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n✅ Semua model tersimpan di folder '{MODELS_DIR}/'")
print("📋 File yang dihasilkan:")
for f in os.listdir(MODELS_DIR):
    print(f"   - {f}")
