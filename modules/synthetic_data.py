import random
import pandas as pd


def generate_synthetic_patients(
    report_type="CBC",
    number_of_patients=10,
    seed=None
):
    """
    Generate fictional synthetic patient data.

    This data is artificially generated and does not represent
    real patients.
    """

    if seed is not None:
        random.seed(seed)

    patients = []

    for i in range(number_of_patients):

        patient = {
            "Patient_ID": f"SYN-{random.randint(10000, 99999)}",
            "Age": random.randint(18, 85),
            "Gender": random.choice(["Male", "Female"])
        }

        if report_type == "CBC":

            patient.update({
                "Hemoglobin": round(random.uniform(9.0, 17.0), 1),
                "RBC": round(random.uniform(3.5, 6.0), 2),
                "WBC": random.randint(4000, 12000),
                "Platelets": random.randint(150000, 450000),
                "Hematocrit": round(random.uniform(30, 52), 1),
                "MCV": round(random.uniform(75, 100), 1)
            })

        elif report_type == "Diabetes":

            patient.update({
                "Fasting_Glucose": random.randint(70, 220),
                "Postprandial_Glucose": random.randint(90, 300),
                "HbA1c": round(random.uniform(4.5, 12.0), 1),
                "Insulin": random.randint(20, 250)
            })

        elif report_type == "Lipid":

            patient.update({
                "Total_Cholesterol": random.randint(130, 320),
                "HDL": random.randint(25, 90),
                "LDL": random.randint(60, 240),
                "Triglycerides": random.randint(60, 400)
            })

        elif report_type == "Kidney":

            patient.update({
                "Creatinine": round(random.uniform(0.5, 3.5), 2),
                "Urea": random.randint(15, 120),
                "eGFR": random.randint(20, 130),
                "Uric_Acid": round(random.uniform(2.5, 10.0), 1)
            })

        patients.append(patient)

    return pd.DataFrame(patients)


def save_synthetic_patients(
    dataframe,
    output_path="datasets/synthetic_patients.csv"
):
    """Save synthetic patient data."""

    dataframe.to_csv(output_path, index=False)

    return output_path