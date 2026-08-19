import os
import traceback
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# MODULE IMPORTS
# ============================================================

from modules.synthetic_data import generate_synthetic_patients
from modules.genai_generator import build_complete_report
from modules.report_generator import generate_pdf

from modules.real_report_processor import (
    extract_report_text,
    clean_extracted_text,
    get_report_statistics,
    detect_medical_parameters,
    extract_medical_values
)

from modules.ml_models import (
    predict_diabetes,
    predict_heart_disease,
    prediction_to_text
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Healthcare Report Generator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "patients": None,
    "selected_patient": None,
    "generated_report": None,
    "pdf_file": None,
    "report_type": "Diabetes",
    "input_mode": None,

    "real_report_text": None,
    "real_report_method": None,
    "real_report_parameters": [],
    "real_report_values": None,
    "verified_real_data": None,
    "real_report_name": None,

    # Disease prediction
    "prediction_result": None,
    "prediction_text": None
}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🏥 Generative AI for Healthcare'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Synthetic & Real Medical Report Generation'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SAFETY NOTICE
# ============================================================

st.warning(
    """
    ⚠️ **Healthcare Safety & Privacy Notice**

    This application is an educational/research prototype.

    • Synthetic data can be used for demonstration.

    • For real reports, only upload documents you are
      authorized to process.

    • Avoid uploading unnecessary personally identifiable
      information.

    • Disease predictions are model probability estimates,
      not medical diagnoses.

    • AI-generated content must not be used as a substitute
      for professional medical diagnosis or treatment.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    report_types = [
        "Diabetes",
        "CBC",
        "Lipid",
        "Kidney"
    ]

    current_index = report_types.index(
        st.session_state.report_type
    )

    report_type = st.selectbox(
        "Medical Report Type",
        report_types,
        index=current_index
    )

    st.session_state.report_type = report_type

    st.divider()

    st.subheader("🤖 GenAI")

    model_name = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-120b"
    )

    st.write(
        f"Model: `{model_name}`"
    )

    if os.getenv("GROQ_API_KEY"):

        st.success(
            "Groq API configured"
        )

    else:

        st.error(
            "GROQ_API_KEY missing"
        )

    st.divider()

    st.caption(
        "Educational / Research Prototype"
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "👤 Synthetic Patients",
        "📄 Real Medical Report",
        "🤖 GenAI Report",
        "📑 PDF Report",
        "🩺 Disease Prediction",
        "ℹ️ About"
    ]
)


# ============================================================
# TAB 1
# SYNTHETIC PATIENTS
# ============================================================

with tab1:

    st.markdown(
        '<div class="section-title">'
        '👤 Synthetic Patient Generation'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        Generate completely fictional patient records
        for testing and demonstration.
        """
    )

    number_of_patients = st.slider(
        "Number of Synthetic Patients",
        1,
        20,
        5
    )

    if st.button(
        "🧬 Generate Synthetic Patients",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Generating synthetic patients..."
        ):

            try:

                # Current synthetic_data.py accepts
                # number_of_patients rather than n_patients.
                patients = generate_synthetic_patients(
                    report_type=report_type,
                    number_of_patients=number_of_patients
                )

                if isinstance(
                    patients,
                    pd.DataFrame
                ):

                    df = patients

                else:

                    df = pd.DataFrame(
                        patients
                    )

                st.session_state.patients = df

                st.session_state.input_mode = (
                    "synthetic"
                )

                st.session_state.generated_report = None

                st.session_state.pdf_file = None

                st.session_state.prediction_result = None

                st.success(
                    "Synthetic patients generated successfully."
                )

            except Exception as e:

                st.error(
                    f"Generation failed: {e}"
                )

                with st.expander(
                    "Technical Details"
                ):

                    st.code(
                        traceback.format_exc()
                    )


    if st.session_state.patients is not None:

        st.subheader(
            "Generated Patients"
        )

        st.dataframe(
            st.session_state.patients,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# TAB 2
# REAL MEDICAL REPORT
# ============================================================

with tab2:

    st.markdown(
        '<div class="section-title">'
        '📄 Real Medical Report Processing'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        Upload a laboratory report in PDF, JPG, JPEG,
        or PNG format.
        """
    )

    st.info(
        """
        **Processing flow**

        Upload → Extract Text → Detect Parameters →
        Extract Values → Verify → AI Report
        """
    )

    st.warning(
        """
        🔐 Do not upload medical records containing
        personally identifiable information unless you
        are authorized to process them.
        """
    )

    uploaded_report = st.file_uploader(
        "Upload Medical Report",
        type=[
            "pdf",
            "png",
            "jpg",
            "jpeg"
        ]
    )

    if uploaded_report is not None:

        st.session_state.real_report_name = (
            uploaded_report.name
        )

        st.success(
            f"Uploaded: {uploaded_report.name}"
        )

        file_size = (
            uploaded_report.size / 1024
        )

        st.write(
            f"File size: {file_size:.2f} KB"
        )

        # ----------------------------------------------------
        # EXTRACT TEXT
        # ----------------------------------------------------

        if st.button(
            "🔍 Extract Report",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Extracting medical report..."
            ):

                try:

                    result = extract_report_text(
                        uploaded_report
                    )

                    raw_text = result.get(
                        "text",
                        ""
                    )

                    cleaned_text = (
                        clean_extracted_text(
                            raw_text
                        )
                    )

                    st.session_state.real_report_text = (
                        cleaned_text
                    )

                    st.session_state.real_report_method = (
                        result.get(
                            "method",
                            "Unknown"
                        )
                    )

                    parameters = (
                        detect_medical_parameters(
                            cleaned_text
                        )
                    )

                    st.session_state.real_report_parameters = (
                        parameters
                    )

                    st.session_state.input_mode = (
                        "real"
                    )

                    st.session_state.generated_report = None

                    st.session_state.pdf_file = None

                    st.session_state.real_report_values = None

                    st.session_state.verified_real_data = None

                    st.session_state.prediction_result = None

                    st.success(
                        "Report extracted successfully."
                    )

                except Exception as e:

                    st.error(
                        "Report extraction failed."
                    )

                    st.code(
                        str(e)
                    )

                    with st.expander(
                        "Technical Details"
                    ):

                        st.code(
                            traceback.format_exc()
                        )


    # --------------------------------------------------------
    # SHOW EXTRACTED REPORT
    # --------------------------------------------------------

    if st.session_state.real_report_text:

        st.divider()

        st.subheader(
            "📋 Extracted Report"
        )

        st.info(
            "Method: "
            + str(
                st.session_state.real_report_method
            )
        )

        edited_text = st.text_area(
            "Review / Correct Extracted Text",
            value=st.session_state.real_report_text,
            height=400
        )

        if st.button(
            "💾 Save Corrected Text",
            use_container_width=True
        ):

            st.session_state.real_report_text = (
                edited_text
            )

            st.session_state.real_report_parameters = (
                detect_medical_parameters(
                    edited_text
                )
            )

            st.session_state.real_report_values = None

            st.session_state.verified_real_data = None

            st.success(
                "Corrected text saved."
            )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        statistics = get_report_statistics(
            st.session_state.real_report_text
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Characters",
                statistics["characters"]
            )

        with col2:

            st.metric(
                "Words",
                statistics["words"]
            )

        with col3:

            st.metric(
                "Lines",
                statistics["lines"]
            )

        # ----------------------------------------------------
        # DETECTED PARAMETERS
        # ----------------------------------------------------

        st.subheader(
            "🔬 Detected Parameters"
        )

        detected = (
            st.session_state.real_report_parameters
        )

        if detected:

            st.success(
                ", ".join(detected)
            )

        else:

            st.warning(
                "No supported laboratory parameters detected."
            )

        # ----------------------------------------------------
        # EXTRACT NUMERIC VALUES
        # ----------------------------------------------------

        if st.button(
            "🧬 Extract Medical Values",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Extracting laboratory values..."
            ):

                try:

                    values = extract_medical_values(
                        st.session_state.real_report_text
                    )

                    st.session_state.real_report_values = (
                        values
                    )

                    st.success(
                        "Medical values extracted."
                    )

                except Exception as e:

                    st.error(
                        f"Value extraction failed: {e}"
                    )

        # ----------------------------------------------------
        # DISPLAY VALUES
        # ----------------------------------------------------

        if st.session_state.real_report_values:

            st.divider()

            st.subheader(
                "🧪 Extracted Laboratory Values"
            )

            values = (
                st.session_state.real_report_values
            )

            value_rows = []

            for parameter, value_data in values.items():

                if isinstance(
                    value_data,
                    dict
                ):

                    value = value_data.get(
                        "value",
                        ""
                    )

                    unit = value_data.get(
                        "unit",
                        ""
                    )

                else:

                    value = value_data

                    unit = ""

                value_rows.append(
                    {
                        "Parameter": parameter,
                        "Value": value,
                        "Unit": unit
                    }
                )

            if value_rows:

                values_df = pd.DataFrame(
                    value_rows
                )

                edited_values_df = st.data_editor(
                    values_df,
                    use_container_width=True,
                    hide_index=True,
                    num_rows="dynamic"
                )

                if st.button(
                    "✅ Confirm Medical Values",
                    type="primary",
                    use_container_width=True
                ):

                    verified_data = {}

                    for _, row in edited_values_df.iterrows():

                        parameter = str(
                            row["Parameter"]
                        ).strip()

                        value = row["Value"]

                        unit = str(
                            row["Unit"]
                        ).strip()

                        if parameter:

                            verified_data[
                                parameter
                            ] = {
                                "value": value,
                                "unit": unit
                            }

                    st.session_state.verified_real_data = (
                        verified_data
                    )

                    st.success(
                        "Medical values verified successfully."
                    )

            # ------------------------------------------------
            # CONFIRMED VALUES
            # ------------------------------------------------

            if st.session_state.verified_real_data:

                st.subheader(
                    "✅ Confirmed Medical Data"
                )

                confirmed_rows = []

                for parameter, data in (
                    st.session_state.verified_real_data.items()
                ):

                    confirmed_rows.append(
                        {
                            "Parameter": parameter,
                            "Value": data["value"],
                            "Unit": data["unit"]
                        }
                    )

                st.dataframe(
                    pd.DataFrame(
                        confirmed_rows
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                st.success(
                    """
                    The extracted values have been verified.
                    You can now generate the AI report or use
                    the Disease Prediction tab.
                    """
                )


# ============================================================
# TAB 3
# GENAI REPORT
# ============================================================

with tab3:

    st.markdown(
        '<div class="section-title">'
        '🤖 Generative AI Report'
        '</div>',
        unsafe_allow_html=True
    )

    # ========================================================
    # REAL REPORT
    # ========================================================

    if (
        st.session_state.input_mode == "real"
        and st.session_state.verified_real_data
    ):

        st.subheader(
            "📄 Real Medical Report"
        )

        verified_data = (
            st.session_state.verified_real_data
        )

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Parameter": parameter,
                        "Value": data["value"],
                        "Unit": data["unit"]
                    }
                    for parameter, data
                    in verified_data.items()
                ]
            ),
            use_container_width=True,
            hide_index=True
        )

        if st.button(
            "🤖 Generate AI Report",
            type="primary",
            use_container_width=True
        ):

            if not os.getenv(
                "GROQ_API_KEY"
            ):

                st.error(
                    "GROQ_API_KEY is not configured."
                )

            else:

                with st.spinner(
                    "Generating AI report..."
                ):

                    try:

                        real_patient = {
                            "Data_Source":
                                "USER-UPLOADED REAL REPORT",

                            "Report_Type":
                                report_type
                        }

                        for parameter, data in (
                            verified_data.items()
                        ):

                            real_patient[
                                parameter
                            ] = data["value"]

                            if data["unit"]:

                                real_patient[
                                    parameter + " Unit"
                                ] = data["unit"]

                        report = build_complete_report(
                            real_patient,
                            report_type
                        )

                        st.session_state.generated_report = (
                            report
                        )

                        st.success(
                            "AI report generated successfully."
                        )

                    except Exception as e:

                        st.error(
                            "GenAI request failed."
                        )

                        st.code(
                            str(e)
                        )

                        with st.expander(
                            "Technical Details"
                        ):

                            st.code(
                                traceback.format_exc()
                            )


    # ========================================================
    # SYNTHETIC REPORT
    # ========================================================

    elif (
        st.session_state.input_mode == "synthetic"
        and st.session_state.patients is not None
    ):

        st.subheader(
            "👤 Synthetic Patient"
        )

        patients_df = (
            st.session_state.patients
        )

        patient_index = st.selectbox(
            "Select Patient",
            options=list(
                range(
                    len(patients_df)
                )
            ),
            format_func=lambda x:
                f"Patient {x + 1}"
        )

        selected_patient = (
            patients_df.iloc[
                patient_index
            ].to_dict()
        )

        st.session_state.selected_patient = (
            selected_patient
        )

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Parameter": key,
                        "Value": value
                    }
                    for key, value
                    in selected_patient.items()
                ]
            ),
            use_container_width=True,
            hide_index=True
        )

        if st.button(
            "🤖 Generate AI Report",
            type="primary",
            use_container_width=True
        ):

            if not os.getenv(
                "GROQ_API_KEY"
            ):

                st.error(
                    "GROQ_API_KEY is not configured."
                )

            else:

                with st.spinner(
                    "Generating AI report..."
                ):

                    try:

                        report = build_complete_report(
                            selected_patient,
                            report_type
                        )

                        st.session_state.generated_report = (
                            report
                        )

                        st.success(
                            "AI report generated successfully."
                        )

                    except Exception as e:

                        st.error(
                            "GenAI request failed."
                        )

                        st.code(
                            str(e)
                        )

    else:

        st.info(
            """
            First generate a synthetic patient or upload
            and verify a real medical report.
            """
        )


    # ========================================================
    # DISPLAY GENERATED REPORT
    # ========================================================

    if st.session_state.generated_report:

        st.divider()

        st.subheader(
            "📋 Generated Medical Report"
        )

        st.markdown(
            st.session_state.generated_report
        )


# ============================================================
# TAB 4
# PDF REPORT
# ============================================================

with tab4:

    st.markdown(
        '<div class="section-title">'
        '📑 PDF Report'
        '</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.generated_report:

        st.info(
            "Generate an AI report first."
        )

    else:

        st.success(
            "AI report is ready for PDF generation."
        )

        if st.button(
            "📄 Create PDF",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Creating PDF..."
            ):

                try:

                    timestamp = (
                        datetime.now().strftime(
                            "%Y%m%d_%H%M%S"
                        )
                    )

                    source = (
                        st.session_state.input_mode
                        or "unknown"
                    )

                    filename = (
                        f"{report_type.lower()}_"
                        f"{source}_"
                        f"medical_report_"
                        f"{timestamp}.pdf"
                    )

                    pdf_path = generate_pdf(
                        st.session_state.generated_report,
                        filename
                    )

                    with open(
                        pdf_path,
                        "rb"
                    ) as f:

                        pdf_bytes = f.read()

                    st.session_state.pdf_file = (
                        pdf_bytes
                    )

                    st.success(
                        "PDF generated successfully."
                    )

                except Exception as e:

                    st.error(
                        "PDF generation failed."
                    )

                    st.code(
                        str(e)
                    )

                    with st.expander(
                        "Technical Details"
                    ):

                        st.code(
                            traceback.format_exc()
                        )

        if st.session_state.pdf_file:

            st.download_button(
                label="⬇️ Download PDF Report",
                data=st.session_state.pdf_file,
                file_name=(
                    f"{report_type.lower()}_"
                    f"medical_report.pdf"
                ),
                mime="application/pdf",
                use_container_width=True
            )


# ============================================================
# TAB 5
# DISEASE PREDICTION
# ============================================================

with tab5:

    st.markdown(
        '<div class="section-title">'
        '🩺 Disease Risk Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.warning(
        """
        ⚠️ This feature provides model probability estimates
        for educational/research purposes only. It is NOT a
        medical diagnosis and should not be used for treatment
        decisions.
        """
    )

    prediction_source = st.radio(
        "Prediction Input",
        [
            "Manual Values",
            "Verified Real Report",
            "Synthetic Patient"
        ],
        horizontal=True
    )

    disease = st.selectbox(
        "Select Disease",
        [
            "Diabetes",
            "Heart Disease"
        ]
    )

    # ========================================================
    # MANUAL VALUES
    # ========================================================

    if prediction_source == "Manual Values":

        st.subheader(
            "🧪 Enter Medical Parameters"
        )

        if disease == "Diabetes":

            col1, col2, col3 = st.columns(3)

            with col1:

                glucose = st.number_input(
                    "Glucose",
                    min_value=0.0,
                    max_value=500.0,
                    value=120.0,
                    step=1.0
                )

            with col2:

                insulin = st.number_input(
                    "Insulin",
                    min_value=0.0,
                    max_value=1000.0,
                    value=100.0,
                    step=1.0
                )

            with col3:

                age = st.number_input(
                    "Age",
                    min_value=1.0,
                    max_value=120.0,
                    value=40.0,
                    step=1.0
                )

            prediction_input = {
                "Glucose": glucose,
                "Insulin": insulin,
                "Age": age
            }

        else:

            col1, col2 = st.columns(2)

            with col1:

                age = st.number_input(
                    "Age",
                    min_value=1.0,
                    max_value=120.0,
                    value=50.0,
                    step=1.0
                )

                trestbps = st.number_input(
                    "Resting Blood Pressure",
                    min_value=50.0,
                    max_value=250.0,
                    value=130.0,
                    step=1.0
                )

                chol = st.number_input(
                    "Cholesterol",
                    min_value=50.0,
                    max_value=700.0,
                    value=220.0,
                    step=1.0
                )

            with col2:

                thalach = st.number_input(
                    "Maximum Heart Rate",
                    min_value=50.0,
                    max_value=250.0,
                    value=150.0,
                    step=1.0
                )

                oldpeak = st.number_input(
                    "Oldpeak",
                    min_value=0.0,
                    max_value=10.0,
                    value=1.0,
                    step=0.1
                )

            prediction_input = {
                "age": age,
                "trestbps": trestbps,
                "chol": chol,
                "thalach": thalach,
                "oldpeak": oldpeak
            }


    # ========================================================
    # VERIFIED REAL REPORT
    # ========================================================

    elif prediction_source == "Verified Real Report":

        st.subheader(
            "📄 Verified Medical Report Values"
        )

        if not st.session_state.verified_real_data:

            st.info(
                """
                Upload a real medical report in the
                Real Medical Report tab, extract the
                values, and confirm them first.
                """
            )

            prediction_input = None

        else:

            verified = (
                st.session_state.verified_real_data
            )

            rows = []

            for parameter, data in verified.items():

                rows.append(
                    {
                        "Parameter": parameter,
                        "Value": data["value"],
                        "Unit": data["unit"]
                    }
                )

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True
            )

            prediction_input = {}

            for parameter, data in verified.items():

                prediction_input[
                    parameter
                ] = data["value"]

            if disease == "Diabetes":

                # Common aliases from medical reports
                aliases = {
                    "Glucose": [
                        "Glucose",
                        "Fasting Glucose",
                        "Fasting_Glucose",
                        "Blood Glucose",
                        "Fasting Blood Glucose"
                    ],
                    "Insulin": [
                        "Insulin"
                    ],
                    "Age": [
                        "Age",
                        "Patient Age"
                    ]
                }

                normalized_input = {}

                for standard_name, possible_names in aliases.items():

                    for name in possible_names:

                        if name in prediction_input:

                            normalized_input[
                                standard_name
                            ] = prediction_input[name]

                            break

                prediction_input = normalized_input

            else:

                aliases = {
                    "age": [
                        "age",
                        "Age",
                        "Patient Age"
                    ],
                    "trestbps": [
                        "trestbps",
                        "Resting Blood Pressure",
                        "Blood Pressure",
                        "Systolic Blood Pressure"
                    ],
                    "chol": [
                        "chol",
                        "Cholesterol",
                        "Total Cholesterol"
                    ],
                    "thalach": [
                        "thalach",
                        "Maximum Heart Rate",
                        "Max Heart Rate"
                    ],
                    "oldpeak": [
                        "oldpeak",
                        "Oldpeak"
                    ]
                }

                normalized_input = {}

                for standard_name, possible_names in aliases.items():

                    for name in possible_names:

                        if name in prediction_input:

                            normalized_input[
                                standard_name
                            ] = prediction_input[name]

                            break

                prediction_input = normalized_input


    # ========================================================
    # SYNTHETIC PATIENT
    # ========================================================

    else:

        st.subheader(
            "👤 Synthetic Patient"
        )

        if st.session_state.patients is None:

            st.info(
                """
                Generate synthetic patients first from
                the Synthetic Patients tab.
                """
            )

            prediction_input = None

        else:

            patients_df = (
                st.session_state.patients
            )

            patient_index = st.selectbox(
                "Select Patient for Prediction",
                options=list(
                    range(
                        len(patients_df)
                    )
                ),
                format_func=lambda x:
                    f"Patient {x + 1}",
                key="prediction_patient_index"
            )

            selected = (
                patients_df.iloc[
                    patient_index
                ].to_dict()
            )

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Parameter": key,
                            "Value": value
                        }
                        for key, value
                        in selected.items()
                    ]
                ),
                use_container_width=True,
                hide_index=True
            )

            prediction_input = selected.copy()

            if disease == "Diabetes":

                # Map synthetic diabetes fields
                if "Fasting_Glucose" in prediction_input:

                    prediction_input["Glucose"] = (
                        prediction_input[
                            "Fasting_Glucose"
                        ]
                    )

                elif "Postprandial_Glucose" in prediction_input:

                    prediction_input["Glucose"] = (
                        prediction_input[
                            "Postprandial_Glucose"
                        ]
                    )

            # Age and Insulin already use compatible names.


    # ========================================================
    # RUN PREDICTION
    # ========================================================

    if prediction_input is not None:

        st.divider()

        if st.button(
            "🔍 Predict Disease Risk",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Running machine learning model..."
            ):

                try:

                    if disease == "Diabetes":

                        result = predict_diabetes(
                            prediction_input
                        )

                    else:

                        result = predict_heart_disease(
                            prediction_input
                        )

                    st.session_state.prediction_result = (
                        result
                    )

                    st.session_state.prediction_text = (
                        prediction_to_text(
                            result
                        )
                    )

                except Exception as e:

                    st.session_state.prediction_result = None

                    st.error(
                        f"Prediction failed: {e}"
                    )

                    with st.expander(
                        "Technical Details"
                    ):

                        st.code(
                            traceback.format_exc()
                        )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    result = st.session_state.prediction_result

    if result:

        st.divider()

        st.subheader(
            "📊 Prediction Result"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Disease",
                result["disease"]
            )

        with col2:

            st.metric(
                "Model Probability",
                f"{result['probability_percent']:.2f}%"
            )

        with col3:

            st.metric(
                "Model",
                result["model"]
            )

        if result["prediction"] == 1:

            st.error(
                f"⚠️ {result['prediction_label']}"
            )

        else:

            st.success(
                f"✅ {result['prediction_label']}"
            )

        st.info(
            f"Probability category: **{result['category']}**"
        )

        st.progress(
            min(
                max(
                    result["probability"],
                    0.0
                ),
                1.0
            )
        )

        st.subheader(
            "🔬 Parameters Used"
        )

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Parameter": key,
                        "Value": value
                    }
                    for key, value
                    in result["features"].items()
                ]
            ),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "The model was trained using "
            f"{result['training_rows']} rows from the "
            "project dataset. This small dataset is "
            "not sufficient for clinical validation."
        )

        # ----------------------------------------------------
        # Add prediction to report
        # ----------------------------------------------------

        if st.button(
            "➕ Add Prediction to AI/PDF Report",
            use_container_width=True
        ):

            prediction_block = (
                "\n\n---\n\n"
                "## 🩺 Machine Learning Risk Prediction\n\n"
                + prediction_to_text(result)
            )

            if st.session_state.generated_report:

                st.session_state.generated_report += (
                    prediction_block
                )

            else:

                st.session_state.generated_report = (
                    prediction_block
                )

            st.success(
                "Prediction added to the report. "
                "Open the GenAI or PDF Report tab."
            )


# ============================================================
# TAB 6
# ABOUT
# ============================================================

with tab6:

    st.markdown(
        '<div class="section-title">'
        'ℹ️ About the Project'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        ## Generative AI for Healthcare

        ### Synthetic & Real Medical Report Generation
        ### + Disease Risk Prediction

        This project demonstrates a healthcare AI
        prototype that combines:

        • Synthetic patient generation

        • Real medical report processing

        • Medical parameter extraction

        • Generative AI report generation

        • Machine learning disease-risk prediction

        • Model probability estimation

        • PDF report generation
        """
    )

    st.subheader(
        "Supported Input Modes"
    )

    st.write(
        "🧪 **Synthetic Data**"
    )

    st.write(
        "Generate artificial patient and laboratory data."
    )

    st.write(
        "📄 **Real Medical Reports**"
    )

    st.write(
        "Upload PDF, JPG, JPEG, or PNG laboratory reports."
    )

    st.write(
        "🩺 **Disease Prediction**"
    )

    st.write(
        "Estimate model probability for diabetes or "
        "heart-disease risk using selected laboratory "
        "and clinical parameters."
    )

    st.subheader(
        "System Architecture"
    )

    st.code(
        """
                  INPUT
                    │
          ┌─────────┴──────────┐
          │                    │
     Synthetic             Real Report
      Patient              PDF / Image
          │                    │
          │              Text Extraction
          │                    │
          │             Medical Parameters
          │                    │
          └─────────┬──────────┘
                    │
             Verified Values
                    │
          ┌─────────┴──────────┐
          │                    │
      ML Models             Groq GenAI
          │                    │
   Disease Probability    AI Explanation
          │                    │
          └─────────┬──────────┘
                    │
             Combined Report
                    │
                 PDF Output
        """,
        language="text"
    )

    st.subheader(
        "Important"
    )

    st.warning(
        """
        This system is an educational/research prototype.

        Disease predictions are model probability estimates
        and are NOT medical diagnoses.

        The current demonstration datasets are very small
        and are not clinically validated.

        Real patient information should be handled according
        to applicable privacy and security requirements.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Generative AI for Healthcare | "
    "Synthetic & Real Medical Report Generation | "
    "Disease Risk Prediction | "
    "Educational / Research Use Only"
)