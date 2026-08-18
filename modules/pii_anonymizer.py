import re


def anonymize_text(text):
    """
    Remove or replace common personally identifiable information.
    """

    if not text:
        return ""

    # Email addresses
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[REDACTED_EMAIL]",
        text
    )

    # Phone numbers
    text = re.sub(
        r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b",
        "[REDACTED_PHONE]",
        text
    )

    # Aadhaar-like numbers
    text = re.sub(
        r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        "[REDACTED_ID]",
        text
    )

    # Patient ID
    text = re.sub(
        r"(?i)(patient\s*(id|number)|patient\s*no)\s*[:\-]?\s*[A-Za-z0-9\-]+",
        "Patient ID: [REDACTED_ID]",
        text
    )

    # Date of birth
    text = re.sub(
        r"(?i)(date\s*of\s*birth|dob)\s*[:\-]?\s*[0-9/\-]+",
        "Date of Birth: [REDACTED_DOB]",
        text
    )

    # Names after common labels
    text = re.sub(
        r"(?i)(patient\s*name|name)\s*[:\-]\s*[A-Za-z .]+",
        "Patient Name: [ANONYMOUS]",
        text
    )

    return text


def anonymize_patient_data(patient_data):
    """
    Anonymize a patient dictionary.
    """

    anonymized = dict(patient_data)

    sensitive_fields = [
        "name",
        "patient_name",
        "phone",
        "email",
        "address",
        "patient_id"
    ]

    for field in sensitive_fields:

        if field in anonymized:
            anonymized[field] = "[REDACTED]"

    if "Patient_ID" in anonymized:
        anonymized["Patient_ID"] = "SYNTHETIC-PATIENT"

    return anonymized