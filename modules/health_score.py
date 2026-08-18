def calculate_health_score(glucose, bmi, age):

    score = 100

    if glucose > 140:
        score -= 30

    if bmi > 30:
        score -= 20

    if age > 50:
        score -= 10

    return score