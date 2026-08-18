def chatbot_response(question):

    q = question.lower()

    # Hemoglobin / anemia
    if "hemoglobin" in q or "anemia" in q or "iron" in q:
        return (
            "Low hemoglobin may indicate anemia. "
            "Iron supplements are sometimes recommended, but you should consult a doctor first. "
            "Eating iron-rich foods like spinach, beans, lentils, red meat, and fortified cereals "
            "can help increase hemoglobin levels."
        )

    # Diabetes / blood sugar
    elif "diabetes" in q or "blood sugar" in q or "glucose" in q:
        return (
            "High blood sugar may indicate diabetes. "
            "Managing diabetes involves regular exercise, healthy eating, monitoring blood sugar, "
            "and following your doctor's advice."
        )

    # Cholesterol
    elif "cholesterol" in q:
        return (
            "High cholesterol increases the risk of heart disease. "
            "Reducing fried foods, exercising regularly, and eating more fiber-rich foods "
            "like oats, fruits, and vegetables can help lower cholesterol."
        )

    # Blood pressure
    elif "blood pressure" in q or "bp" in q:
        return (
            "High blood pressure can increase the risk of heart disease and stroke. "
            "Reducing salt intake, exercising regularly, managing stress, and maintaining "
            "a healthy weight can help control blood pressure."
        )

    # Exercise
    elif "exercise" in q or "workout" in q:
        return (
            "Regular exercise helps improve heart health, control weight, and manage blood sugar. "
            "At least 30 minutes of moderate exercise daily is recommended."
        )

    # Diet / food
    elif "diet" in q or "food" in q or "eat" in q:
        return (
            "A healthy diet should include fruits, vegetables, whole grains, lean proteins, "
            "and healthy fats. Avoid excessive sugar, salt, and processed foods."
        )

    # Heart disease
    elif "heart" in q:
        return (
            "Heart disease risk increases with high cholesterol, high blood pressure, "
            "smoking, diabetes, and obesity. Healthy lifestyle choices can significantly "
            "reduce this risk."
        )

    # Default response
    else:
        return (
            "I can answer questions about diabetes, cholesterol, hemoglobin, blood pressure, "
            "diet, exercise, and heart health."
        )