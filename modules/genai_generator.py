import os
import re

from dotenv import load_dotenv
from groq import Groq

from modules.medical_templates import (
    get_report_title,
    get_report_parameters,
    get_reference_range
)

# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-120b"
def get_groq_api_key():
    # Streamlit Cloud Secrets
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    # Local .env
    return os.getenv("GROQ_API_KEY")


GROQ_API_KEY = get_groq_api_key()


def get_groq_client():
    if not GROQ_API_KEY:
        return None

    return Groq(api_key=GROQ_API_KEY)


# ============================================================
# GROQ CLIENT
# ============================================================

def create_groq_client():
    """
    Create and return a Groq API client.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured.\n"
            "Please add your Groq API key to the .env file."
        )

    return Groq(api_key=api_key)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

def get_model_name():
    """
    Return the Groq model configured in .env.
    """

    return os.getenv(
        "GROQ_MODEL",
        DEFAULT_MODEL
    )


# ============================================================
# LABORATORY DATA EXTRACTION
# ============================================================

def get_laboratory_values(patient, report_type):
    """
    Extract all laboratory parameters supplied by
    the synthetic patient generator.

    This function does NOT use the LLM.
    Therefore, laboratory values cannot be changed
    or invented by the LLM at this stage.
    """

    parameters = get_report_parameters(
        report_type
    )

    laboratory_values = []

    for parameter in parameters:

        if parameter not in patient:
            continue

        value = patient[parameter]

        reference = get_reference_range(
            parameter
        )

        laboratory_values.append(
            {
                "parameter": parameter,
                "value": value,
                "reference": reference
            }
        )

    return laboratory_values


# ============================================================
# FORMAT LABORATORY DATA FOR THE PROMPT
# ============================================================

def format_laboratory_data(
    laboratory_values
):
    """
    Convert laboratory values into a controlled
    text block for the LLM.
    """

    lines = []

    for index, item in enumerate(
        laboratory_values,
        start=1
    ):

        parameter = item["parameter"]
        value = item["value"]
        reference = item["reference"]

        lines.append(
            f"{index}. "
            f"{parameter} = {value} | "
            f"Reference information = {reference}"
        )

    return "\n".join(lines)


# ============================================================
# BUILD AI PROMPT
# ============================================================

def build_medical_prompt(
    patient,
    report_type
):
    """
    Build a controlled healthcare prompt.

    The LLM is responsible only for interpretation
    and narrative sections.

    Python remains responsible for the factual
    laboratory values.
    """

    report_title = get_report_title(
        report_type
    )

    laboratory_values = get_laboratory_values(
        patient,
        report_type
    )

    laboratory_text = format_laboratory_data(
        laboratory_values
    )

    parameter_names = [
        item["parameter"]
        for item in laboratory_values
    ]

    parameter_list = ", ".join(
        parameter_names
    )

    prompt = f"""
You are a careful Generative AI assistant
for a healthcare education and research
application.

You are generating a SYNTHETIC medical report.

THIS IS NOT A REAL PATIENT.

==================================================
REPORT INFORMATION
==================================================

Report Type:
{report_title}

Patient ID:
{patient.get("Patient_ID", "SYNTHETIC-PATIENT")}

Age:
{patient.get("Age", "Not provided")}

Gender:
{patient.get("Gender", "Not provided")}

==================================================
LABORATORY DATA PROVIDED BY THE APPLICATION
==================================================

{laboratory_text}

==================================================
MANDATORY DATA-INTEGRITY RULES
==================================================

The following laboratory parameters were supplied:

{parameter_list}

You MUST discuss EVERY ONE of these parameters.

CRITICAL RULES:

1. Create exactly ONE interpretation bullet
   for EVERY laboratory parameter supplied.

2. Do NOT omit any supplied parameter.

3. Do NOT change any supplied laboratory value.

4. Do NOT round, modify, or replace supplied values.

5. Do NOT invent laboratory values.

6. Do NOT invent laboratory units.

7. Do NOT invent reference ranges.

8. If a reference range is unavailable, explicitly
   state that the laboratory-specific reference
   range was not provided.

9. Keep the parameters in the same order in which
   they were supplied.

10. Do NOT say that a parameter was unavailable
    or not supplied if it appears above.

11. Do NOT introduce additional patient laboratory
    values.

12. Do NOT create a diagnosis.

13. Do NOT prescribe medication.

14. Do NOT provide medication dosage.

15. Do NOT recommend treatment as a definitive
    clinical instruction.

16. Use cautious educational language.

17. Clearly distinguish observations from diagnosis.

18. If information is missing, say:
    "This information was not provided."

19. The patient is synthetic.

20. The final report is for educational and
    research purposes only.

==================================================
REQUIRED OUTPUT
==================================================

Generate exactly these sections:

### 1. General Interpretation

Create exactly ONE bullet for EACH supplied
laboratory parameter.

Each bullet must:

- Name the parameter.
- Include its exact supplied value.
- Provide a cautious general interpretation.
- Mention reference-range limitations where applicable.

Example structure:

- Parameter Name: exact value — cautious
  educational interpretation.

Do NOT skip any parameter.

--------------------------------------------------

### 2. Observations

Provide observations based ONLY on:

- Patient age
- Patient gender
- Supplied laboratory values

Do not claim that a supplied parameter is missing.

Do not invent symptoms, medical history,
medications, diagnoses, or other test results.

--------------------------------------------------

### 3. Possible Areas for Professional Review

Provide general areas that a qualified healthcare
professional may consider reviewing.

Do not provide a diagnosis.

Do not prescribe treatment.

Do not recommend medication dosage.

--------------------------------------------------

### 4. Suggested Questions for a Healthcare Professional

Provide 4–5 useful questions that could be
discussed with a qualified healthcare professional.

--------------------------------------------------

### 5. Safety Disclaimer

Clearly state:

- The data are synthetic.
- The report is for education/research only.
- It is not a real clinical record.
- It is not a substitute for professional
  medical advice, diagnosis, or treatment.

==================================================
FINAL CHECK BEFORE RESPONDING
==================================================

Before returning the report, verify:

- Every supplied parameter has an interpretation.
- Every supplied value is unchanged.
- No supplied parameter was omitted.
- No new laboratory values were invented.
- No diagnosis was presented as a fact.
- No medication prescription was provided.
- The synthetic-data disclaimer is present.

Return ONLY the five requested sections.
"""

    return prompt


# ============================================================
# GENERATE AI REPORT
# ============================================================

def generate_medical_report(
    patient,
    report_type
):
    """
    Generate the AI narrative using Groq.
    """

    client = create_groq_client()

    model = get_model_name()

    prompt = build_medical_prompt(
        patient,
        report_type
    )

    response = client.chat.completions.create(

        model=model,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful healthcare "
                    "Generative AI assistant. "
                    "You generate synthetic medical "
                    "reports for educational and "
                    "research purposes. "
                    "Preserve all supplied laboratory "
                    "values exactly."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.1,

        max_completion_tokens=2000
    )

    return response.choices[0].message.content


# ============================================================
# DATA-INTEGRITY CHECK
# ============================================================

def check_report_data_integrity(
    report,
    patient,
    report_type
):
    """
    Verify that every supplied laboratory parameter
    and value appears in the AI-generated report.

    Returns a list of missing items.
    """

    laboratory_values = get_laboratory_values(
        patient,
        report_type
    )

    report_text = str(report)

    missing_items = []

    for item in laboratory_values:

        parameter = str(
            item["parameter"]
        )

        value = str(
            item["value"]
        )

        parameter_present = (
            parameter.lower()
            in report_text.lower()
        )

        value_present = (
            value
            in report_text
        )

        if not parameter_present:
            missing_items.append(
                f"{parameter} (parameter missing)"
            )

        elif not value_present:
            missing_items.append(
                f"{parameter} ({value} missing)"
            )

    return missing_items


# ============================================================
# EXTRACT NUMBER FROM VALUE
# ============================================================

def normalize_value(value):
    """
    Convert a value into a simple string representation
    for integrity checking.
    """

    if value is None:
        return ""

    return str(value).strip()


# ============================================================
# BUILD FACTUAL LABORATORY TABLE
# ============================================================

def build_laboratory_table(
    patient,
    report_type
):
    """
    Build the laboratory table directly from Python.

    IMPORTANT:
    The LLM does NOT generate this table.
    """

    laboratory_values = get_laboratory_values(
        patient,
        report_type
    )

    lines = []

    lines.append(
        "| Parameter | Value | Reference Information |"
    )

    lines.append(
        "|---|---:|---|"
    )

    for item in laboratory_values:

        parameter = item["parameter"]

        value = normalize_value(
            item["value"]
        )

        reference = item["reference"]

        lines.append(
            f"| {parameter} | {value} | {reference} |"
        )

    return "\n".join(lines)


# ============================================================
# BUILD COMPLETE REPORT
# ============================================================

def build_complete_report(
    patient,
    report_type
):
    """
    Build the complete synthetic medical report.

    Python:
        - Patient information
        - Laboratory values
        - Laboratory table
        - Data-integrity validation

    Groq:
        - Interpretation
        - Observations
        - Professional review areas
        - Questions
        - Disclaimer
    """

    report_title = get_report_title(
        report_type
    )

    laboratory_values = get_laboratory_values(
        patient,
        report_type
    )

    # --------------------------------------------------------
    # Generate AI narrative
    # --------------------------------------------------------

    ai_report = generate_medical_report(
        patient,
        report_type
    )

    # --------------------------------------------------------
    # Check data integrity
    # --------------------------------------------------------

    missing_items = check_report_data_integrity(
        ai_report,
        patient,
        report_type
    )

    # --------------------------------------------------------
    # If values are missing, retry once with a strict prompt
    # --------------------------------------------------------

    if missing_items:

        retry_prompt = f"""
The previous AI response failed a data-integrity
check.

The following supplied laboratory parameters or
values were missing:

{", ".join(missing_items)}

You MUST correct this.

Generate the report sections again.

Every supplied laboratory parameter MUST appear.

Every supplied value MUST appear exactly.

Do not omit any parameter.

Do not invent any new values.

Do not modify any value.

Use cautious educational medical language.

This is synthetic data only.

Return:

### 1. General Interpretation
### 2. Observations
### 3. Possible Areas for Professional Review
### 4. Suggested Questions for a Healthcare Professional
### 5. Safety Disclaimer
"""

        client = create_groq_client()

        response = client.chat.completions.create(

            model=get_model_name(),

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a healthcare "
                        "Generative AI assistant. "
                        "You must preserve all "
                        "laboratory parameters "
                        "and values exactly."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        build_medical_prompt(
                            patient,
                            report_type
                        )
                        + "\n\n"
                        + retry_prompt
                    )
                }
            ],

            temperature=0.0,

            max_completion_tokens=2000
        )

        ai_report = (
            response
            .choices[0]
            .message
            .content
        )

    # --------------------------------------------------------
    # Build final report
    # --------------------------------------------------------

    report = []

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    report.append(
        "# SYNTHETIC MEDICAL REPORT"
    )

    report.append("")

    report.append(
        f"**Report Type:** {report_title}"
    )

    report.append("")

    # --------------------------------------------------------
    # Synthetic patient information
    # --------------------------------------------------------

    report.append(
        "## Synthetic Patient Information"
    )

    report.append("")

    report.append(
        f"- **Patient ID:** "
        f"{patient.get('Patient_ID', 'SYNTHETIC-PATIENT')}"
    )

    report.append(
        f"- **Age:** "
        f"{patient.get('Age', 'Not provided')}"
    )

    report.append(
        f"- **Gender:** "
        f"{patient.get('Gender', 'Not provided')}"
    )

    report.append(
        f"- **Report Type:** "
        f"{report_title}"
    )

    report.append("")

    # --------------------------------------------------------
    # Laboratory results
    # --------------------------------------------------------

    report.append(
        "## Laboratory Values"
    )

    report.append("")

    report.append(
        build_laboratory_table(
            patient,
            report_type
        )
    )

    report.append("")

    # --------------------------------------------------------
    # AI-generated report
    # --------------------------------------------------------

    report.append(
        "## Generative AI Report"
    )

    report.append("")

    report.append(
        ai_report
    )

    report.append("")

    # --------------------------------------------------------
    # Data integrity notice
    # --------------------------------------------------------

    final_missing_items = (
        check_report_data_integrity(
            ai_report,
            patient,
            report_type
        )
    )

    if final_missing_items:

        report.append(
            "## Data Integrity Notice"
        )

        report.append("")

        report.append(
            "The AI narrative did not explicitly "
            "mention the following supplied "
            "laboratory parameters: "
            + ", ".join(
                final_missing_items
            )
        )

        report.append("")

    else:

        report.append(
            "## Data Integrity Status"
        )

        report.append("")

        report.append(
            "All supplied laboratory parameters "
            "and values were verified against the "
            "AI-generated narrative."
        )

        report.append("")

    # --------------------------------------------------------
    # Final safety disclaimer
    # --------------------------------------------------------

    report.append(
        "---"
    )

    report.append("")

    report.append(
        "**IMPORTANT DISCLAIMER:** This document "
        "contains synthetically generated medical "
        "data and AI-generated educational content. "
        "It does not represent a real patient, is "
        "not a clinical record, and must not be used "
        "for diagnosis, treatment, or medical "
        "decision-making. Laboratory reference "
        "ranges can vary between laboratories. "
        "Interpretation of medical information "
        "should be performed by a qualified "
        "healthcare professional."
    )

    return "\n".join(report)


# ============================================================
# SIMPLE COMPATIBILITY FUNCTION
# ============================================================

def generate_report(
    patient,
    report_type
):
    """
    Compatibility wrapper.

    Use this if main.py currently calls
    generate_report().
    """

    return build_complete_report(
        patient,
        report_type
    )