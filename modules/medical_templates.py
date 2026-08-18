REPORT_TYPES = {
    "CBC": {
        "title": "Complete Blood Count",
        "parameters": [
            "Hemoglobin",
            "RBC",
            "WBC",
            "Platelets",
            "Hematocrit",
            "MCV"
        ]
    },

    "Diabetes": {
        "title": "Diabetes Profile",
        "parameters": [
            "Fasting Glucose",
            "Postprandial Glucose",
            "HbA1c",
            "Insulin"
        ]
    },

    "Lipid": {
        "title": "Lipid Profile",
        "parameters": [
            "Total Cholesterol",
            "HDL",
            "LDL",
            "Triglycerides"
        ]
    },

    "Kidney": {
        "title": "Kidney Function Profile",
        "parameters": [
            "Creatinine",
            "Urea",
            "eGFR",
            "Uric Acid"
        ]
    }
}


REFERENCE_RANGES = {

    "Hemoglobin": "Approximately 12–17 g/dL; varies by sex, age and laboratory",

    "RBC": "Approximately 4.0–6.0 million/µL",

    "WBC": "Approximately 4,000–11,000 cells/µL",

    "Platelets": "Approximately 150,000–450,000/µL",

    "Hematocrit": "Approximately 36–52%",

    "MCV": "Approximately 80–100 fL",

    "Fasting Glucose": "Common adult reference range approximately 70–99 mg/dL",

    "HbA1c": "Below 5.7% is commonly considered normal",

    "Total Cholesterol": "Below 200 mg/dL is generally desirable",

    "HDL": "Higher values are generally more favorable",

    "LDL": "Lower values are generally more favorable",

    "Triglycerides": "Below 150 mg/dL is generally desirable",

    "Creatinine": "Typical adult reference range varies by sex and laboratory",

    "Urea": "Reference range varies by laboratory",

    "eGFR": "90 or above is generally considered normal if no other kidney abnormalities are present",

    "Uric Acid": "Reference range varies by sex and laboratory"
}


def get_report_parameters(report_type):
    """Return parameters for a report type."""

    return REPORT_TYPES.get(
        report_type,
        {}
    ).get("parameters", [])


def get_report_title(report_type):
    """Return title for a report."""

    return REPORT_TYPES.get(
        report_type,
        {}
    ).get("title", "Synthetic Medical Report")


def get_reference_range(parameter):
    """Return reference information."""

    return REFERENCE_RANGES.get(
        parameter,
        "Reference range varies by laboratory."
    )