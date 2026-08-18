import re


PATTERNS = {

    "Hemoglobin": [
        r"(?i)hemoglobin\s*[:\-]?\s*(\d+(?:\.\d+)?)",
        r"(?i)hb\s*[:\-]?\s*(\d+(?:\.\d+)?)"
    ],

    "RBC": [
        r"(?i)rbc\s*[:\-]?\s*(\d+(?:\.\d+)?)"
    ],

    "WBC": [
        r"(?i)wbc\s*[:\-]?\s*(\d+(?:\.\d+)?)"
    ],

    "Platelets": [
        r"(?i)platelets?\s*[:\-]?\s*(\d+(?:\.\d+)?)"
    ],

    "Blood Sugar": [
        r"(?i)blood\s*sugar\s*[:\-]?\s*(\d+(?:\.\d+)?)",
        r"(?i)glucose\s*[:\-]?\s*(\d+(?:\.\d+)?)"
    ],

    "Fasting Glucose": [
        r"(?i)fasting\s*(?:blood\s*)?glucose\s*[:\-]?\s*(\d+(?:\.\d+)?)"
    ],

    "HbA1c": [
        r"(?i)hba1c\s*[:\-]?\s*(\d+(?:\.\d+)?)"
    ],

    "Total Cholesterol": [
        r"(?i)total\s*cholesterol\s*[:\-]?\s*(\d+(?:\.\d+)?)"
    ],

    "HDL": [
        r"(?i)hdl\s*[:\-]?\s*(\d+(?:\.\d+)?)"
    ],

    "LDL": [
        r"(?i)ldl\s*[:\-]?\s*(\d+(?:\.\d+)?)"
    ],

    "Triglycerides": [
        r"(?i)triglycerides?\s*[:\-]?\s*(\d+(?:\.\d+)?)"
    ],

    "Creatinine": [
        r"(?i)creatinine\s*[:\-]?\s*(\d+(?:\.\d+)?)"
    ],

    "Urea": [
        r"(?i)urea\s*[:\-]?\s*(\d+(?:\.\d+)?)"
    ],

    "eGFR": [
        r"(?i)egfr\s*[:\-]?\s*(\d+(?:\.\d+)?)"
    ],

    "Uric Acid": [
        r"(?i)uric\s*acid\s*[:\-]?\s*(\d+(?:\.\d+)?)"
    ]
}


def extract_lab_values(text):
    """
    Extract laboratory values from report text.
    """

    values = {}

    if not text:
        return values

    for name, patterns in PATTERNS.items():

        for pattern in patterns:

            match = re.search(pattern, text)

            if match:

                try:
                    values[name] = float(match.group(1))
                except ValueError:
                    pass

                break

    return values