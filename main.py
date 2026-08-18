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

    "real_report_name": None
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

    .small-text {
        font-size: 13px;
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

    • Synthetic data can be used freely for demonstration.

    • For real reports, only upload documents you are
      authorized to process.

    • Avoid uploading unnecessary personally identifiable
      information.
      
    • AI-generated content must not be used as a substitute
      for professional medical diagnosis or treatment.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    report_type = st.selectbox(
        "Medical Report Type",
        [
            "Diabetes",
            "CBC",
            "Lipid",
            "Kidney"
        ],
        index=[
            "Diabetes",
            "CBC",
            "Lipid",
            "Kidney"
        ].index(
            st.session_state.report_type
        )
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

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "👤 Synthetic Patients",
        "📄 Real Medical Report",
        "🤖 GenAI Report",
        "📑 PDF Report",
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
        Generate completely synthetic patient records
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

                patients = generate_synthetic_patients(
                    report_type=report_type,
                    n_patients=number_of_patients
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

                st.success(
                    "Synthetic patients generated successfully."
                )

            except TypeError:

                try:

                    patients = generate_synthetic_patients(
                        report_type,
                        number_of_patients
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

                    st.success(
                        "Synthetic patients generated successfully."
                    )

                except Exception as e:

                    st.error(
                        f"Generation failed: {e}"
                    )

                    st.code(
                        traceback.format_exc()
                    )

            except Exception as e:

                st.error(
                    f"Generation failed: {e}"
                )

                st.code(
                    traceback.format_exc()
                )


    # --------------------------------------------------------
    # DISPLAY PATIENTS
    # --------------------------------------------------------

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
        Extract Values → Verify → Generate AI Report
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

        # ----------------------------------------------------
        # Editable extracted text
        # ----------------------------------------------------

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

            parameter_text = ", ".join(
                detected
            )

            st.success(
                parameter_text
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

                # ------------------------------------------------
                # Save verified values
                # ------------------------------------------------

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

            # ----------------------------------------------------
            # Confirmed values
            # ----------------------------------------------------

            if st.session_state.verified_real_data:

                st.subheader(
                    "✅ Confirmed Medical Data"
                )

                confirmed_rows = []

                for parameter, data in (
                    st.session_state
                    .verified_real_data
                    .items()
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
                    The extracted values have been
                    verified. You can now generate
                    the AI report from the GenAI tab.
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

        st.success(
            "Verified medical values are ready."
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
            First generate a synthetic patient or
            upload and verify a real medical report.
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


        # ----------------------------------------------------
        # Download
        # ----------------------------------------------------

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
# ABOUT
# ============================================================

with tab5:

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

        This project demonstrates a Generative AI
        pipeline for processing medical laboratory
        information and generating structured
        educational reports.
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

    st.subheader(
        "System Architecture"
    )

    st.code(
        """
             INPUT
               │
       ┌───────┴────────┐
       │                │
  Synthetic          Real Report
       │                │
       │           PDF / Image
       │                │
       │         Text Extraction
       │                │
       │              OCR
       │                │
       └───────┬────────┘
               │
       Medical Parameters
               │
        Value Extraction
               │
       User Verification
               │
          Groq GenAI
               │
       AI Medical Report
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
        The system is not a medical diagnostic tool.

        AI-generated interpretations should not be
        considered professional medical advice.

        Real patient information should be handled
        according to applicable privacy and security
        requirements.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Generative AI for Healthcare | "
    "Synthetic & Real Medical Report Generation | "
    "Educational / Research Use Only"
)