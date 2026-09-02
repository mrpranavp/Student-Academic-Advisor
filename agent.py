import time
import streamlit as st
from google import genai
from google.genai import types

from statistical_tools import (
    predict_student_risk,
    analyse_risk_drivers
)


# Statistical risk prediction tool
def risk_prediction_tool(
    hours_studied: float,
    attendance: float,
    sleep_hours: float,
    previous_scores: float,
    tutoring_sessions: int,
    physical_activity: float
) -> dict:
    """
    Predict a student's academic risk using the trained
    Linear Discriminant Analysis model.
    """

    profile = {
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Sleep_Hours": sleep_hours,
        "Previous_Scores": previous_scores,
        "Tutoring_Sessions": tutoring_sessions,
        "Physical_Activity": physical_activity
    }

    result = predict_student_risk(profile)

    return {
        "classification": result["classification"],
        "risk_probability": round(
            result["risk_probability"] * 100,
            2
        )
    }


# LDA driver analysis tool
def risk_driver_tool(
    hours_studied: float,
    attendance: float,
    sleep_hours: float,
    previous_scores: float,
    tutoring_sessions: int,
    physical_activity: float
) -> dict:
    """
    Identify the student's most influential profile
    characteristics according to the trained LDA model.
    """

    profile = {
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Sleep_Hours": sleep_hours,
        "Previous_Scores": previous_scores,
        "Tutoring_Sessions": tutoring_sessions,
        "Physical_Activity": physical_activity
    }

    drivers = analyse_risk_drivers(profile).head(3)

    output = []

    for _, row in drivers.iterrows():
        contribution = float(row["Contribution"])

        output.append({
            "variable": row["Variable"],
            "student_value": float(row["Student_Value"]),
            "lda_contribution": round(contribution, 3),
            "direction": (
                "toward At Risk"
                if contribution > 0
                else "away from At Risk"
            )
        })

    return {
        "top_drivers": output
    }


# Academic support tool
def academic_support_tool(
    hours_studied: float,
    attendance: float,
    sleep_hours: float,
    previous_scores: float,
    tutoring_sessions: int,
    physical_activity: float
) -> dict:
    """
    Generate practical academic support options based
    on the student's observed profile.
    """

    recommendations = []

    if attendance < 75:
        recommendations.append(
            "Work on improving attendance and identify barriers causing absence."
        )

    if hours_studied < 15:
        recommendations.append(
            "Develop a structured weekly study schedule."
        )

    if previous_scores < 65:
        recommendations.append(
            "Review weaker academic areas and seek targeted support."
        )

    if tutoring_sessions == 0:
        recommendations.append(
            "Consider tutoring or additional academic-support sessions."
        )

    if sleep_hours < 6:
        recommendations.append(
            "Maintain a more consistent sleep routine to support learning and concentration."
        )

    if physical_activity == 0:
        recommendations.append(
            "Consider including regular physical activity as part of a balanced routine."
        )

    if not recommendations:
        recommendations.append(
            "Maintain the current academic routine and continue monitoring performance."
        )

    return {
        "recommendations": recommendations
    }


# Gemini Agent
def run_student_agent(
    student_profile: dict,
    user_question: str
) -> str:

    client = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )

    prompt = f"""
You are an academic decision-support AI agent.

A student has provided the following profile:

Hours Studied: {student_profile["Hours_Studied"]}
Attendance: {student_profile["Attendance"]}%
Sleep Hours: {student_profile["Sleep_Hours"]}
Previous Scores: {student_profile["Previous_Scores"]}
Tutoring Sessions: {student_profile["Tutoring_Sessions"]}
Physical Activity: {student_profile["Physical_Activity"]}

The user asks:

"{user_question}"

You have access to statistical and academic-support tools.

For questions about this student's risk, ALWAYS use
risk_prediction_tool.

When explaining why the student received a classification,
use risk_driver_tool.

When the user asks what the student can improve or what
support may help, use academic_support_tool.

You may use multiple tools when appropriate.

IMPORTANT:
- Never invent a risk probability.
- Risk probabilities must come from risk_prediction_tool.
- Do not claim that predictors cause academic performance.
- Describe them as statistical associations or model characteristics.
- When discussing LDA contributions, distinguish whether a
  characteristic pushes toward or away from the At Risk class.
- Explain results in clear language suitable for a student.
- The model is a decision-support demonstration, not an
  official institutional failure classification.
"""

    # Retry Gemini automatically if the service is temporarily overloaded
    for attempt in range(3):

        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[
                        risk_prediction_tool,
                        risk_driver_tool,
                        academic_support_tool
                    ],
                    temperature=0.2
                )
            )

            return response.text

        except Exception as error:

            error_message = str(error)

            if (
                "503" in error_message
                or "UNAVAILABLE" in error_message
            ):
                if attempt < 2:
                    time.sleep(3)
                    continue

            raise error
