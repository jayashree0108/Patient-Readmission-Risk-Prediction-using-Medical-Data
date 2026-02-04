import pandas as pd
import numpy as np

def generate_patient_data(n=5000, seed=42):
    np.random.seed(seed)

    data = pd.DataFrame({
        "age": np.random.randint(20, 90, n),
        "gender": np.random.choice(["Male", "Female"], n),
        "num_prior_admissions": np.random.poisson(2, n),
        "length_of_stay": np.random.randint(1, 15, n),
        "num_lab_procedures": np.random.randint(1, 100, n),
        "num_medications": np.random.randint(1, 25, n),
        "has_diabetes": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        "has_hypertension": np.random.choice([0, 1], n, p=[0.6, 0.4]),
        "emergency_admission": np.random.choice([0, 1], n, p=[0.75, 0.25]),
        "discharge_to_home": np.random.choice([0, 1], n, p=[0.8, 0.2])
    })

    # Risk logic for readmission
    risk_score = (
        0.03 * data["age"] +
        0.8 * data["num_prior_admissions"] +
        0.5 * data["length_of_stay"] +
        0.4 * data["num_medications"] +
        1.5 * data["has_diabetes"] +
        1.2 * data["has_hypertension"] +
        1.0 * data["emergency_admission"] -
        1.3 * data["discharge_to_home"]
    )

    probability = 1 / (1 + np.exp(-0.05 * risk_score))
    data["readmitted_30_days"] = np.random.binomial(1, probability)

    return data

if __name__ == "__main__":
    df = generate_patient_data()
    df.to_csv("data/patient_data.csv", index=False)
    print("Dataset generated!")