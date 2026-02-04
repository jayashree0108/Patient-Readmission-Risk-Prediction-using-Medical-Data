import pandas as pd
import numpy as np
import joblib
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ------------------ Strong Synthetic Clinical Dataset ------------------
def generate_patient_data(n=8000, seed=42):
    np.random.seed(seed)

    df = pd.DataFrame({
        "age": np.random.randint(35, 90, n),
        "gender": np.random.choice(["Male", "Female"], n),
        "num_prior_admissions": np.random.poisson(2, n),
        "length_of_stay": np.random.randint(1, 20, n),
        "num_lab_procedures": np.random.randint(10, 120, n),
        "num_medications": np.random.randint(1, 30, n),
        "has_diabetes": np.random.choice([0, 1], n, p=[0.72, 0.28]),
        "has_hypertension": np.random.choice([0, 1], n, p=[0.68, 0.32]),
        "emergency_admission": np.random.choice([0, 1], n, p=[0.78, 0.22]),
        "discharge_to_home": np.random.choice([0, 1], n, p=[0.82, 0.18])
    })

    # Clinical risk score with stronger separation
    risk_score = (
        0.08 * df["age"] +
        1.8 * df["num_prior_admissions"] +
        1.2 * df["length_of_stay"] +
        0.9 * df["num_medications"] +
        3.0 * df["has_diabetes"] +
        2.5 * df["has_hypertension"] +
        2.2 * df["emergency_admission"] -
        3.2 * df["discharge_to_home"]
    )

    prob = 1 / (1 + np.exp(-0.06 * risk_score))
    df["readmitted_30_days"] = np.random.binomial(1, prob)

    return df

# ------------------ Data Preparation ------------------
df = generate_patient_data()

le = LabelEncoder()
df["gender"] = le.fit_transform(df["gender"])

X = df.drop("readmitted_30_days", axis=1)
y = df["readmitted_30_days"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# ------------------ XGBoost Model ------------------
model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.04,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, "model.pkl")

# ------------------ SHAP (Now Works) ------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)