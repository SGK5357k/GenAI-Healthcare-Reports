def explain_lab_results(values):

    explanations = []

    hb = values.get("Hemoglobin")
    sugar = values.get("Blood Sugar")
    chol = values.get("Cholesterol")

    if hb and hb < 12:
        explanations.append("Your hemoglobin level is low, which may indicate anemia or low red blood cell count.")

    if sugar and sugar > 140:
        explanations.append("Your blood sugar level is high, which may indicate a risk of diabetes.")

    if chol and chol > 200:
        explanations.append("Your cholesterol level is high, which may increase the risk of heart disease.")

    return explanations