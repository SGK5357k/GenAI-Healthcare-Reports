"""
modules/ml_models.py

Healthcare Disease Risk Prediction
----------------------------------

Supported:
    1. Diabetes
    2. Heart Disease

Features:
    - Partial patient input is allowed.
    - Missing values are automatically imputed using
      training-data median values.
    - Model confidence is returned.
    - Missing parameters are reported.

IMPORTANT:
This system is intended for educational/research use.
It is NOT a medical diagnostic system.
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "datasets"

DIABETES_DATASET = DATASET_DIR / "diabetes.csv"
HEART_DATASET = DATASET_DIR / "heart.csv"


# ============================================================
# MODEL CACHE
# ============================================================

_DIABETES_MODEL = None
_HEART_MODEL = None


# ============================================================
# DIABETES FEATURES
# ============================================================

DIABETES_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]


# ============================================================
# HEART FEATURES
# ============================================================

HEART_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]


# ============================================================
# FEATURE ALIASES
# ============================================================

ALIASES = {

    # ---------------- DIABETES ----------------

    "Pregnancies": [
        "Pregnancies",
        "pregnancies",
        "Pregnancy",
        "pregnancy"
    ],

    "Glucose": [
        "Glucose",
        "glucose",
        "Fasting Glucose",
        "Fasting_Glucose",
        "Blood Glucose",
        "Blood_Glucose"
    ],

    "BloodPressure": [
        "BloodPressure",
        "bloodpressure",
        "Blood Pressure",
        "blood_pressure",
        "BP"
    ],

    "SkinThickness": [
        "SkinThickness",
        "Skin Thickness",
        "skin_thickness"
    ],

    "Insulin": [
        "Insulin",
        "insulin"
    ],

    "BMI": [
        "BMI",
        "bmi",
        "Body Mass Index",
        "Body_Mass_Index"
    ],

    "DiabetesPedigreeFunction": [
        "DiabetesPedigreeFunction",
        "Diabetes Pedigree Function",
        "diabetes_pedigree_function",
        "DPF"
    ],

    "Age": [
        "Age",
        "age",
        "Patient Age",
        "patient_age"
    ],

    # ---------------- HEART ----------------

    "sex": [
        "sex",
        "Sex",
        "gender",
        "Gender"
    ],

    "cp": [
        "cp",
        "CP",
        "chest pain",
        "Chest Pain",
        "ChestPain"
    ],

    "trestbps": [
        "trestbps",
        "Trestbps",
        "Resting Blood Pressure",
        "Blood Pressure",
        "BP"
    ],

    "chol": [
        "chol",
        "Chol",
        "Cholesterol",
        "Total Cholesterol"
    ],

    "fbs": [
        "fbs",
        "FBS",
        "Fasting Blood Sugar"
    ],

    "restecg": [
        "restecg",
        "RestECG",
        "Resting ECG",
        "Resting Electrocardiographic Result"
    ],

    "thalach": [
        "thalach",
        "Thalach",
        "Maximum Heart Rate",
        "Max Heart Rate"
    ],

    "exang": [
        "exang",
        "Exang",
        "Exercise Induced Angina",
        "Exercise-Induced Angina"
    ],

    "oldpeak": [
        "oldpeak",
        "Oldpeak",
        "Old Peak"
    ],

    "slope": [
        "slope",
        "Slope"
    ],

    "ca": [
        "ca",
        "CA"
    ],

    "thal": [
        "thal",
        "Thal"
    ]
}


# ============================================================
# VALUE CONVERSION
# ============================================================

def _to_float(value):
    """
    Convert a value to float.

    Examples:
        120
        "120"
        "120 mg/dL"
        "120.5"

    Invalid or empty values become NaN.
    """

    if value is None:
        return np.nan

    if isinstance(
        value,
        (int, float, np.integer, np.floating)
    ):

        if pd.isna(value):
            return np.nan

        return float(value)

    text = str(value).strip()

    if not text:
        return np.nan

    # Handle common Streamlit empty values
    if text.lower() in [
        "none",
        "null",
        "nan",
        "na",
        "n/a",
        "-"
    ]:

        return np.nan

    text = text.replace(",", "")

    match = re.search(
        r"-?\d+(?:\.\d+)?",
        text
    )

    if match:

        try:
            return float(match.group())

        except ValueError:

            return np.nan

    return np.nan


# ============================================================
# GET VALUE FROM INPUT
# ============================================================

def _get_value(data, feature):
    """
    Find a feature from the supplied dictionary.

    Supports multiple naming styles.
    """

    if data is None:
        return np.nan

    if isinstance(data, pd.Series):

        data = data.to_dict()

    if not isinstance(data, dict):

        return np.nan

    possible_names = ALIASES.get(
        feature,
        [feature]
    )

    # --------------------------------------------------------
    # Exact matching
    # --------------------------------------------------------

    for name in possible_names:

        if name in data:

            value = _to_float(
                data[name]
            )

            if not np.isnan(value):

                return value

    # --------------------------------------------------------
    # Case-insensitive matching
    # --------------------------------------------------------

    lower_data = {
        str(key).strip().lower(): value
        for key, value in data.items()
    }

    for name in possible_names:

        key = str(name).strip().lower()

        if key in lower_data:

            value = _to_float(
                lower_data[key]
            )

            if not np.isnan(value):

                return value

    return np.nan


# ============================================================
# PREPARE INPUT
# ============================================================

def _prepare_input(data, features):
    """
    Prepare patient data.

    IMPORTANT:
    Missing values are intentionally kept as NaN.
    They will be handled by the trained imputer.
    """

    if isinstance(data, pd.Series):

        data = data.to_dict()

    if not isinstance(data, dict):

        raise TypeError(
            "Prediction input must be a dictionary "
            "or pandas Series."
        )

    values = {}

    for feature in features:

        values[feature] = _get_value(
            data,
            feature
        )

    return pd.DataFrame(
        [values],
        columns=features
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def _train_model(
    dataset_path,
    features,
    target_column
):
    """
    Train a Random Forest model from a CSV dataset.

    Missing training values are handled with median
    imputation.
    """

    if not dataset_path.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n"
            f"{dataset_path}\n\n"
            f"Make sure the dataset exists inside:\n"
            f"{DATASET_DIR}"
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        dataset_path
    )

    # --------------------------------------------------------
    # Validate target
    # --------------------------------------------------------

    if target_column not in df.columns:

        raise ValueError(
            f"\nTarget column '{target_column}' "
            f"was not found in {dataset_path.name}.\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    # --------------------------------------------------------
    # Validate features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            f"\nMissing feature columns in "
            f"{dataset_path.name}:\n"
            f"{missing_features}\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    # --------------------------------------------------------
    # Prepare X and y
    # --------------------------------------------------------

    X = df[
        features
    ].copy()

    y = df[
        target_column
    ].copy()

    # --------------------------------------------------------
    # Convert X to numeric
    # --------------------------------------------------------

    for column in features:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Convert target
    # --------------------------------------------------------

    y = pd.to_numeric(
        y,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove invalid target rows
    # --------------------------------------------------------

    valid_rows = y.notna()

    X = X.loc[
        valid_rows
    ].reset_index(
        drop=True
    )

    y = y.loc[
        valid_rows
    ].astype(int).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Check target classes
    # --------------------------------------------------------

    if y.nunique() < 2:

        raise ValueError(
            f"The target column '{target_column}' "
            "must contain at least two classes."
        )

    # --------------------------------------------------------
    # Imputer
    # --------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X_imputed = imputer.fit_transform(
        X
    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=1,
        n_jobs=-1
    )

    model.fit(
        X_imputed,
        y
    )

    # --------------------------------------------------------
    # Training information
    # --------------------------------------------------------

    training_info = {

        "model": model,

        "imputer": imputer,

        "features": features,

        "training_rows": len(X),

        "target_column": target_column,

        "dataset": dataset_path.name
    }

    return training_info


# ============================================================
# GET DIABETES MODEL
# ============================================================

def get_diabetes_model():

    global _DIABETES_MODEL

    if _DIABETES_MODEL is None:

        _DIABETES_MODEL = _train_model(

            dataset_path=DIABETES_DATASET,

            features=DIABETES_FEATURES,

            target_column="Outcome"
        )

    return _DIABETES_MODEL


# ============================================================
# GET HEART MODEL
# ============================================================

def get_heart_model():

    global _HEART_MODEL

    if _HEART_MODEL is None:

        _HEART_MODEL = _train_model(

            dataset_path=HEART_DATASET,

            features=HEART_FEATURES,

            target_column="target"
        )

    return _HEART_MODEL


# ============================================================
# PROBABILITY CATEGORY
# ============================================================

def probability_category(
    probability
):
    """
    Presentation category for model probability.

    This is NOT a clinical risk classification.
    """

    probability = float(
        probability
    )

    if probability < 0.30:

        return "Lower model probability"

    elif probability < 0.60:

        return "Intermediate model probability"

    elif probability < 0.80:

        return "Higher model probability"

    else:

        return "Very high model probability"


# ============================================================
# GET MISSING FEATURES
# ============================================================

def _get_missing_features(
    X,
    features
):
    """
    Return the list of features that were not supplied
    by the user.
    """

    missing = []

    for feature in features:

        value = X.iloc[0][feature]

        if pd.isna(value):

            missing.append(
                feature
            )

    return missing


# ============================================================
# DIABETES PREDICTION
# ============================================================

def predict_diabetes(data):
    """
    Predict diabetes using diabetes.csv.

    Partial input is supported.

    Example:

        {
            "Glucose": 140,
            "BMI": 28.5,
            "Age": 35
        }

    Missing parameters are automatically imputed using
    median values from the training dataset.
    """

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model_info = get_diabetes_model()

    model = model_info["model"]

    imputer = model_info["imputer"]

    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    X = _prepare_input(
        data,
        DIABETES_FEATURES
    )

    # --------------------------------------------------------
    # Find missing features BEFORE imputation
    # --------------------------------------------------------

    missing_features = _get_missing_features(
        X,
        DIABETES_FEATURES
    )

    provided_features = [
        feature
        for feature in DIABETES_FEATURES
        if feature not in missing_features
    ]

    # --------------------------------------------------------
    # Impute missing values
    # --------------------------------------------------------

    X_imputed = imputer.transform(
        X
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = int(
        model.predict(
            X_imputed
        )[0]
    )

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_imputed
    )[0]

    classes = list(
        model.classes_
    )

    # Probability of positive class
    if 1 in classes:

        positive_index = classes.index(1)

        disease_probability = float(
            probabilities[
                positive_index
            ]
        )

    else:

        disease_probability = 0.0

    # Overall model confidence
    confidence = float(
        np.max(
            probabilities
        )
    )

    # --------------------------------------------------------
    # Prediction label
    # --------------------------------------------------------

    if prediction == 1:

        prediction_label = (
            "Higher model-predicted diabetes risk"
        )

    else:

        prediction_label = (
            "Lower model-predicted diabetes risk"
        )

    # --------------------------------------------------------
    # Imputed values
    # --------------------------------------------------------

    imputed_values = {}

    for index, feature in enumerate(
        DIABETES_FEATURES
    ):

        if feature in missing_features:

            imputed_values[
                feature
            ] = float(
                X_imputed[
                    0,
                    index
                ]
            )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "disease":
            "Diabetes",

        "prediction":
            prediction,

        "prediction_label":
            prediction_label,

        "probability":
            disease_probability,

        "probability_percent":
            round(
                disease_probability * 100,
                2
            ),

        "confidence":
            round(
                confidence * 100,
                2
            ),

        "category":
            probability_category(
                disease_probability
            ),

        "risk_level":
            probability_category(
                disease_probability
            ),

        "features": {
            feature: float(
                X.iloc[0][feature]
            )
            if not pd.isna(
                X.iloc[0][feature]
            )
            else None
            for feature in DIABETES_FEATURES
        },

        "provided_features":
            provided_features,

        "missing_features":
            missing_features,

        "imputed_values":
            imputed_values,

        "training_rows":
            model_info[
                "training_rows"
            ],

        "dataset":
            model_info[
                "dataset"
            ],

        "model":
            "Random Forest",

        "partial_input":
            len(missing_features) > 0
    }


# ============================================================
# HEART DISEASE PREDICTION
# ============================================================

def predict_heart_disease(data):
    """
    Predict heart disease using heart.csv.

    Partial input is supported.

    Missing parameters are automatically imputed.
    """

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model_info = get_heart_model()

    model = model_info["model"]

    imputer = model_info["imputer"]

    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    X = _prepare_input(
        data,
        HEART_FEATURES
    )

    # --------------------------------------------------------
    # Find missing parameters
    # --------------------------------------------------------

    missing_features = _get_missing_features(
        X,
        HEART_FEATURES
    )

    provided_features = [
        feature
        for feature in HEART_FEATURES
        if feature not in missing_features
    ]

    # --------------------------------------------------------
    # Impute
    # --------------------------------------------------------

    X_imputed = imputer.transform(
        X
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = int(
        model.predict(
            X_imputed
        )[0]
    )

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_imputed
    )[0]

    classes = list(
        model.classes_
    )

    if 1 in classes:

        positive_index = classes.index(1)

        disease_probability = float(
            probabilities[
                positive_index
            ]
        )

    else:

        disease_probability = 0.0

    confidence = float(
        np.max(
            probabilities
        )
    )

    # --------------------------------------------------------
    # Prediction label
    # --------------------------------------------------------

    if prediction == 1:

        prediction_label = (
            "Higher model-predicted heart disease risk"
        )

    else:

        prediction_label = (
            "Lower model-predicted heart disease risk"
        )

    # --------------------------------------------------------
    # Imputed values
    # --------------------------------------------------------

    imputed_values = {}

    for index, feature in enumerate(
        HEART_FEATURES
    ):

        if feature in missing_features:

            imputed_values[
                feature
            ] = float(
                X_imputed[
                    0,
                    index
                ]
            )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "disease":
            "Heart Disease",

        "prediction":
            prediction,

        "prediction_label":
            prediction_label,

        "probability":
            disease_probability,

        "probability_percent":
            round(
                disease_probability * 100,
                2
            ),

        "confidence":
            round(
                confidence * 100,
                2
            ),

        "category":
            probability_category(
                disease_probability
            ),

        "risk_level":
            probability_category(
                disease_probability
            ),

        "features": {
            feature: float(
                X.iloc[0][feature]
            )
            if not pd.isna(
                X.iloc[0][feature]
            )
            else None
            for feature in HEART_FEATURES
        },

        "provided_features":
            provided_features,

        "missing_features":
            missing_features,

        "imputed_values":
            imputed_values,

        "training_rows":
            model_info[
                "training_rows"
            ],

        "dataset":
            model_info[
                "dataset"
            ],

        "model":
            "Random Forest",

        "partial_input":
            len(missing_features) > 0
    }


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def predict_heart(data):
    """
    Alias for older versions of main.py.
    """

    return predict_heart_disease(
        data
    )


# ============================================================
# GENERIC DISEASE PREDICTION
# ============================================================

def predict_disease(
    disease,
    data
):
    """
    Generic disease prediction function.
    """

    disease_name = str(
        disease
    ).strip().lower()

    if disease_name == "diabetes":

        return predict_diabetes(
            data
        )

    if disease_name in [
        "heart",
        "heart disease",
        "heart_disease"
    ]:

        return predict_heart_disease(
            data
        )

    raise ValueError(
        "Unsupported disease. "
        "Use 'diabetes' or 'heart disease'."
    )


# ============================================================
# RESULT TO TEXT
# ============================================================

def prediction_to_text(result):
    """
    Convert prediction result into text.

    Useful for:
        - GenAI explanation
        - PDF report
        - Streamlit display
    """

    lines = []

    lines.append(
        "## Machine Learning Disease Risk Prediction"
    )

    lines.append("")

    lines.append(
        f"**Disease:** {result['disease']}"
    )

    lines.append(
        f"**Model:** {result['model']}"
    )

    lines.append(
        f"**Prediction:** {result['prediction_label']}"
    )

    lines.append(
        f"**Model Probability:** "
        f"{result['probability_percent']:.2f}%"
    )

    lines.append(
        f"**Model Confidence:** "
        f"{result['confidence']:.2f}%"
    )

    lines.append(
        f"**Probability Category:** "
        f"{result['category']}"
    )

    lines.append("")

    # --------------------------------------------------------
    # Input completeness
    # --------------------------------------------------------

    total_features = len(
        result["features"]
    )

    provided_count = len(
        result["provided_features"]
    )

    lines.append(
        f"**Input Parameters Provided:** "
        f"{provided_count}/{total_features}"
    )

    if result["partial_input"]:

        lines.append("")

        lines.append(
            "⚠️ **Partial Input:** "
            "Some parameters were not provided."
        )

        lines.append("")

        lines.append(
            "**Missing Parameters:** "
            + ", ".join(
                result["missing_features"]
            )
        )

        lines.append("")

        lines.append(
            "Missing parameters were estimated "
            "using median values from the training "
            "dataset."
        )

    lines.append("")

    # --------------------------------------------------------
    # Input parameters
    # --------------------------------------------------------

    lines.append(
        "### Patient Parameters"
    )

    for feature, value in result[
        "features"
    ].items():

        if value is not None:

            lines.append(
                f"- **{feature}:** {value}"
            )

        else:

            lines.append(
                f"- **{feature}:** "
                "Not provided"
            )

    # --------------------------------------------------------
    # Training information
    # --------------------------------------------------------

    lines.append("")

    lines.append(
        "### Model Information"
    )

    lines.append(
        f"- **Dataset:** "
        f"{result['dataset']}"
    )

    lines.append(
        f"- **Training Records:** "
        f"{result['training_rows']}"
    )

    lines.append(
        f"- **Algorithm:** "
        f"{result['model']}"
    )

    # --------------------------------------------------------
    # Disclaimer
    # --------------------------------------------------------

    lines.append("")

    lines.append(
        "⚠️ **Disclaimer:** This prediction is "
        "generated by a machine-learning model for "
        "educational/research purposes only. It is "
        "not a medical diagnosis and should not be "
        "used as a substitute for evaluation by a "
        "qualified healthcare professional."
    )

    return "\n".join(
        lines
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "Testing Healthcare ML Prediction Module"
    )

    print("=" * 60)

    # ========================================================
    # TEST 1: PARTIAL DIABETES INPUT
    # ========================================================

    print(
        "\n\n1. Testing partial diabetes input..."
    )

    diabetes_sample = {

        "Glucose": 140,

        "BMI": 28.5,

        "Age": 35
    }

    try:

        result = predict_diabetes(
            diabetes_sample
        )

        print(
            prediction_to_text(
                result
            )
        )

    except Exception as error:

        print(
            "Diabetes prediction error:"
        )

        print(error)

    # ========================================================
    # TEST 2: FULL DIABETES INPUT
    # ========================================================

    print(
        "\n\n2. Testing complete diabetes input..."
    )

    diabetes_full = {

        "Pregnancies": 2,

        "Glucose": 140,

        "BloodPressure": 80,

        "SkinThickness": 25,

        "Insulin": 100,

        "BMI": 28.5,

        "DiabetesPedigreeFunction": 0.50,

        "Age": 35
    }

    try:

        result = predict_diabetes(
            diabetes_full
        )

        print(
            prediction_to_text(
                result
            )
        )

    except Exception as error:

        print(
            "Diabetes prediction error:"
        )

        print(error)

    # ========================================================
    # TEST 3: PARTIAL HEART INPUT
    # ========================================================

    print(
        "\n\n3. Testing partial heart input..."
    )

    heart_sample = {

        "age": 55,

        "chol": 220,

        "trestbps": 130
    }

    try:

        result = predict_heart_disease(
            heart_sample
        )

        print(
            prediction_to_text(
                result
            )
        )

    except Exception as error:

        print(
            "Heart disease prediction error:"
        )

        print(error)

    print(
        "\n\nTesting completed."
    )