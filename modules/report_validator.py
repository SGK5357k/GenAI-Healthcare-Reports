RANGES = {

    "Age": (0, 120),

    "Hemoglobin": (5, 25),

    "RBC": (2, 8),

    "WBC": (1000, 30000),

    "Platelets": (50000, 1000000),

    "Hematocrit": (15, 70),

    "MCV": (50, 130),

    "Fasting_Glucose": (40, 500),

    "Postprandial_Glucose": (40, 600),

    "HbA1c": (3, 20),

    "Insulin": (0, 1000),

    "Total_Cholesterol": (50, 500),

    "HDL": (10, 150),

    "LDL": (20, 400),

    "Triglycerides": (20, 1000),

    "Creatinine": (0.1, 15),

    "Urea": (1, 300),

    "eGFR": (1, 150),

    "Uric_Acid": (1, 20)
}


def validate_patient(patient):
    """
    Validate one synthetic patient.
    """

    errors = []

    for field, limits in RANGES.items():

        if field not in patient:
            continue

        try:
            value = float(patient[field])
        except (ValueError, TypeError):
            errors.append(
                f"{field}: invalid numeric value"
            )
            continue

        minimum, maximum = limits

        if not minimum <= value <= maximum:

            errors.append(
                f"{field}: value {value} outside allowed range "
                f"{minimum}-{maximum}"
            )

    return errors


def validate_dataframe(dataframe):
    """
    Validate complete synthetic dataset.
    """

    results = []

    for index, row in dataframe.iterrows():

        errors = validate_patient(row.to_dict())

        results.append({
            "row": index + 1,
            "valid": len(errors) == 0,
            "errors": errors
        })

    return results


def is_valid_patient(patient):
    """Return True if patient passes validation."""

    return len(validate_patient(patient)) == 0