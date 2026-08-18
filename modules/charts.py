import streamlit as st
import matplotlib.pyplot as plt


def display_patient_chart(patient):
    """
    Display laboratory values of a synthetic patient.
    """

    values = {}

    excluded_fields = {
        "Patient_ID",
        "Age",
        "Gender"
    }

    for key, value in patient.items():

        if key in excluded_fields:
            continue

        try:
            numeric_value = float(value)
            values[key] = numeric_value
        except (ValueError, TypeError):
            continue

    if not values:
        st.info("No numerical laboratory values available.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        list(values.keys()),
        list(values.values())
    )

    ax.set_title("Synthetic Patient Laboratory Values")
    ax.set_xlabel("Laboratory Parameter")
    ax.set_ylabel("Value")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


def display_dataset_summary(dataframe):
    """
    Display distribution of a selected numerical parameter.
    """

    st.subheader("📊 Synthetic Dataset Summary")

    numerical_columns = dataframe.select_dtypes(
        include="number"
    ).columns.tolist()

    if not numerical_columns:
        st.info("No numerical columns available.")
        return

    selected_column = st.selectbox(
        "Select laboratory parameter",
        numerical_columns
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(
        dataframe[selected_column].dropna(),
        bins=10
    )

    ax.set_title(
        f"Distribution of {selected_column}"
    )

    ax.set_xlabel(selected_column)
    ax.set_ylabel("Number of Synthetic Patients")

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)