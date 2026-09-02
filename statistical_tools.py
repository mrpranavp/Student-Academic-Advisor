import joblib
import pandas as pd

lda_model = joblib.load("lda_model.pkl")
scaler = joblib.load("scaler.pkl")

predictor_vars = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity"
]


def predict_student_risk(student_profile):

    student = pd.DataFrame(
        [student_profile],
        columns=predictor_vars
    )

    scaled_array = scaler.transform(student)

    scaled_student = pd.DataFrame(
        scaled_array,
        columns=predictor_vars
    )

    prediction = int(
        lda_model.predict(scaled_student)[0]
    )

    probability = float(
        lda_model.predict_proba(scaled_student)[0, 1]
    )

    classification = (
        "At Risk"
        if prediction == 1
        else "Not At Risk"
    )

    return {
        "classification": classification,
        "risk_probability": probability
    }


def analyse_risk_drivers(student_profile):

    training_means = pd.Series(
        scaler.mean_,
        index=predictor_vars
    )

    training_stds = pd.Series(
        scaler.scale_,
        index=predictor_vars
    )

    coefficients = pd.Series(
        lda_model.coef_[0],
        index=predictor_vars
    )

    student = pd.Series(student_profile)

    standardized = (
        student[predictor_vars]
        - training_means
    ) / training_stds

    contributions = standardized * coefficients

    result = pd.DataFrame({
        "Variable": predictor_vars,
        "Student_Value": [
            student[v]
            for v in predictor_vars
        ],
        "Contribution": [
            contributions[v]
            for v in predictor_vars
        ]
    })

    result["Absolute_Contribution"] = (
        result["Contribution"].abs()
    )

    return result.sort_values(
        "Absolute_Contribution",
        ascending=False
    )
