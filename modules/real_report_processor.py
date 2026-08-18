import io
import os
import re

import fitz
from PIL import Image


# ============================================================
# SUPPORTED FILE TYPES
# ============================================================

SUPPORTED_EXTENSIONS = [
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg"
]


# ============================================================
# VALIDATE FILE
# ============================================================

def validate_file(uploaded_file):

    if uploaded_file is None:

        return False, "No file uploaded."

    filename = uploaded_file.name.lower()

    extension = os.path.splitext(
        filename
    )[1]

    if extension not in SUPPORTED_EXTENSIONS:

        return (
            False,
            "Unsupported file type."
        )

    return True, "Valid file."


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(uploaded_file):

    file_bytes = uploaded_file.getvalue()

    document = fitz.open(
        stream=file_bytes,
        filetype="pdf"
    )

    pages = []

    for page in document:

        text = page.get_text(
            "text"
        )

        if text:

            pages.append(
                text
            )

    document.close()

    return "\n".join(
        pages
    ).strip()


# ============================================================
# IMAGE OCR
# ============================================================

def extract_text_from_image(uploaded_file):

    try:

        import pytesseract

    except ImportError:

        raise RuntimeError(
            "pytesseract is not installed."
        )

    image = Image.open(
        io.BytesIO(
            uploaded_file.getvalue()
        )
    )

    image = image.convert(
        "RGB"
    )

    text = pytesseract.image_to_string(
        image
    )

    return text.strip()


# ============================================================
# SCANNED PDF OCR
# ============================================================

def extract_text_from_scanned_pdf(
    uploaded_file
):

    try:

        import pytesseract

    except ImportError:

        raise RuntimeError(
            "pytesseract is not installed."
        )

    file_bytes = uploaded_file.getvalue()

    document = fitz.open(
        stream=file_bytes,
        filetype="pdf"
    )

    pages = []

    for page in document:

        pix = page.get_pixmap(
            matrix=fitz.Matrix(
                2,
                2
            )
        )

        image_bytes = pix.tobytes(
            "png"
        )

        image = Image.open(
            io.BytesIO(
                image_bytes
            )
        )

        image = image.convert(
            "RGB"
        )

        text = pytesseract.image_to_string(
            image
        )

        if text:

            pages.append(
                text
            )

    document.close()

    return "\n".join(
        pages
    ).strip()


# ============================================================
# MAIN EXTRACTION
# ============================================================

def extract_report_text(
    uploaded_file
):

    valid, message = validate_file(
        uploaded_file
    )

    if not valid:

        raise ValueError(
            message
        )

    filename = uploaded_file.name.lower()

    extension = os.path.splitext(
        filename
    )[1]

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if extension == ".pdf":

        text = extract_text_from_pdf(
            uploaded_file
        )

        if len(text.strip()) >= 30:

            return {
                "text": text,
                "method": "PDF Text Extraction",
                "ocr_used": False
            }

        try:

            ocr_text = extract_text_from_scanned_pdf(
                uploaded_file
            )

            return {
                "text": ocr_text,
                "method": "OCR - Scanned PDF",
                "ocr_used": True
            }

        except Exception as e:

            return {
                "text": text,
                "method": "PDF Text Extraction",
                "ocr_used": False,
                "ocr_error": str(e)
            }

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    if extension in [
        ".png",
        ".jpg",
        ".jpeg"
    ]:

        text = extract_text_from_image(
            uploaded_file
        )

        return {
            "text": text,
            "method": "OCR - Image",
            "ocr_used": True
        }

    raise ValueError(
        "Unsupported file type."
    )


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_extracted_text(text):

    if not text:

        return ""

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# REPORT STATISTICS
# ============================================================

def get_report_statistics(text):

    if not text:

        return {
            "characters": 0,
            "words": 0,
            "lines": 0
        }

    return {

        "characters": len(text),

        "words": len(
            text.split()
        ),

        "lines": len(
            [
                line
                for line in text.splitlines()
                if line.strip()
            ]
        )
    }


# ============================================================
# MEDICAL PARAMETER PATTERNS
# ============================================================

PARAMETER_PATTERNS = {

    "Fasting Glucose": [
        r"fasting\s+glucose",
        r"fasting\s+blood\s+sugar",
        r"\bfbs\b"
    ],

    "Postprandial Glucose": [
        r"postprandial\s+glucose",
        r"post\s*prandial\s+glucose",
        r"\bppbs\b"
    ],

    "Random Glucose": [
        r"random\s+glucose",
        r"random\s+blood\s+sugar",
        r"\brbs\b"
    ],

    "HbA1c": [
        r"hba1c",
        r"hb\s*a1c",
        r"glycated\s+haemoglobin",
        r"glycated\s+hemoglobin"
    ],

    "Insulin": [
        r"\binsulin\b"
    ],

    "Hemoglobin": [
        r"hemoglobin",
        r"haemoglobin",
        r"\bhb\b"
    ],

    "WBC": [
        r"\bwbc\b",
        r"white\s+blood\s+cell"
    ],

    "RBC": [
        r"\brbc\b",
        r"red\s+blood\s+cell"
    ],

    "Platelets": [
        r"platelet",
        r"platelets"
    ],

    "Total Cholesterol": [
        r"total\s+cholesterol"
    ],

    "LDL": [
        r"\bldl\b"
    ],

    "HDL": [
        r"\bhdl\b"
    ],

    "Triglycerides": [
        r"triglycerides?"
    ],

    "Creatinine": [
        r"\bcreatinine\b"
    ],

    "Urea": [
        r"\burea\b",
        r"blood\s+urea"
    ],

    "eGFR": [
        r"\begfr\b",
        r"estimated\s+glomerular"
    ]
}


# ============================================================
# DETECT PARAMETERS
# ============================================================

def detect_medical_parameters(text):

    detected = []

    text_lower = text.lower()

    for parameter, patterns in PARAMETER_PATTERNS.items():

        for pattern in patterns:

            if re.search(
                pattern,
                text_lower
            ):

                detected.append(
                    parameter
                )

                break

    return detected


# ============================================================
# NUMBER PATTERN
# ============================================================

NUMBER_PATTERN = (
    r"(-?\d+(?:\.\d+)?)"
)


# ============================================================
# EXTRACT VALUE AFTER PARAMETER
# ============================================================

def extract_value_near_parameter(
    text,
    parameter,
    patterns
):

    lines = text.splitlines()

    # --------------------------------------------------------
    # Search line-by-line
    # --------------------------------------------------------

    for line in lines:

        line_clean = line.strip()

        if not line_clean:

            continue

        for pattern in patterns:

            match_parameter = re.search(
                pattern,
                line_clean,
                re.IGNORECASE
            )

            if not match_parameter:

                continue

            # Remove parameter name
            # and search remaining text
            # for numerical value.

            remaining = (
                line_clean[
                    match_parameter.end():
                ]
            )

            number_match = re.search(
                NUMBER_PATTERN,
                remaining
            )

            if number_match:

                value = number_match.group(
                    1
                )

                # Try to capture unit

                after_number = (
                    remaining[
                        number_match.end():
                    ]
                ).strip()

                unit_match = re.match(
                    r"([a-zA-Zµμ/%]+(?:\s*/\s*[a-zA-Zµμ]+)?)",
                    after_number
                )

                unit = ""

                if unit_match:

                    unit = unit_match.group(
                        1
                    )

                return {
                    "value": value,
                    "unit": unit
                }

    return None


# ============================================================
# EXTRACT ALL MEDICAL VALUES
# ============================================================

def extract_medical_values(text):

    extracted = {}

    for parameter, patterns in PARAMETER_PATTERNS.items():

        result = extract_value_near_parameter(
            text,
            parameter,
            patterns
        )

        if result is not None:

            extracted[
                parameter
            ] = result

    return extracted