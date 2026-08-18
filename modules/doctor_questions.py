def doctor_questions(values):

    questions = []

    if "Blood Sugar" in values:
        questions.append("Do I need medication to control my blood sugar?")

    if "Cholesterol" in values:
        questions.append("What diet changes should I make to lower cholesterol?")

    if "Hemoglobin" in values:
        questions.append("Do I need iron supplements for low hemoglobin?")

    questions.append("How often should I repeat these tests?")

    return questions