import json
import time
import streamlit as st
from openai import OpenAI

from statistical_tools import (
    predict_student_risk,
    analyse_risk_drivers
)


#stats risk prediction
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


#lda driver
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
            "student_value": float(
                row["Student_Value"]
            ),
            "lda_contribution": round(
                contribution,
                3
            ),
            "direction": (
                "toward At Risk"
                if contribution > 0
                else "away from At Risk"
            )
        })

    return {
        "top_drivers": output
    }


#support
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


#openrouter tool definitions
tools = [

    {
        "type": "function",

        "function": {

            "name": "risk_prediction_tool",

            "description":
                "Predict the student's academic risk "
                "classification and probability using "
                "the trained LDA model.",

            "parameters": {

                "type": "object",

                "properties": {

                    "hours_studied": {
                        "type": "number"
                    },

                    "attendance": {
                        "type": "number"
                    },

                    "sleep_hours": {
                        "type": "number"
                    },

                    "previous_scores": {
                        "type": "number"
                    },

                    "tutoring_sessions": {
                        "type": "integer"
                    },

                    "physical_activity": {
                        "type": "number"
                    }
                },

                "required": [
                    "hours_studied",
                    "attendance",
                    "sleep_hours",
                    "previous_scores",
                    "tutoring_sessions",
                    "physical_activity"
                ]
            }
        }
    },


    {
        "type": "function",

        "function": {

            "name": "risk_driver_tool",

            "description":
                "Identify the three student characteristics "
                "with the largest LDA contributions to "
                "the student's classification.",

            "parameters": {

                "type": "object",

                "properties": {

                    "hours_studied": {
                        "type": "number"
                    },

                    "attendance": {
                        "type": "number"
                    },

                    "sleep_hours": {
                        "type": "number"
                    },

                    "previous_scores": {
                        "type": "number"
                    },

                    "tutoring_sessions": {
                        "type": "integer"
                    },

                    "physical_activity": {
                        "type": "number"
                    }
                },

                "required": [
                    "hours_studied",
                    "attendance",
                    "sleep_hours",
                    "previous_scores",
                    "tutoring_sessions",
                    "physical_activity"
                ]
            }
        }
    },


    {
        "type": "function",

        "function": {

            "name": "academic_support_tool",

            "description":
                "Generate practical academic support "
                "recommendations based on the student's profile.",

            "parameters": {

                "type": "object",

                "properties": {

                    "hours_studied": {
                        "type": "number"
                    },

                    "attendance": {
                        "type": "number"
                    },

                    "sleep_hours": {
                        "type": "number"
                    },

                    "previous_scores": {
                        "type": "number"
                    },

                    "tutoring_sessions": {
                        "type": "integer"
                    },

                    "physical_activity": {
                        "type": "number"
                    }
                },

                "required": [
                    "hours_studied",
                    "attendance",
                    "sleep_hours",
                    "previous_scores",
                    "tutoring_sessions",
                    "physical_activity"
                ]
            }
        }
    }
]


#tool executor
def execute_tool(
    tool_name,
    arguments
):

    if tool_name == "risk_prediction_tool":
        return risk_prediction_tool(
            **arguments
        )

    if tool_name == "risk_driver_tool":
        return risk_driver_tool(
            **arguments
        )

    if tool_name == "academic_support_tool":
        return academic_support_tool(
            **arguments
        )

    return {
        "error": "Unknown tool"
    }


#openrouter ai agent
def run_student_agent(
    student_profile: dict,
    user_question: str
) -> str:

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=st.secrets["OPENROUTER_API_KEY"]
    )


    #agent instructions
    system_prompt = """
You are an academic decision-support AI agent.

You have access to statistical tools based on a trained
Linear Discriminant Analysis model.

For questions about this student's academic risk,
ALWAYS use the risk_prediction_tool.

When explaining why the student received a classification,
use the risk_driver_tool.

When the user asks what the student can improve or what
support may help, use the academic_support_tool.

You may call multiple tools when appropriate.

IMPORTANT:

- Never invent a risk probability.

- Risk probabilities must come from
  risk_prediction_tool.

- Do not claim that predictors cause academic performance.

- Describe predictors as statistical associations or
  characteristics used by the model.

- When discussing LDA contributions, explain whether
  the contribution points toward At Risk or away from
  At Risk.

- Do not treat every large contribution as negative.
  A large contribution may be protective and point
  away from At Risk.

- Explain results clearly in student-friendly language.

- Recommendations are academic-support suggestions,
  not proven causal interventions.

- This model is a decision-support demonstration and
  not an official institutional failure classification.
"""


    #student information sent to agent
    user_prompt = f"""
Student profile:

Hours Studied:
{student_profile["Hours_Studied"]}

Attendance:
{student_profile["Attendance"]}%

Sleep Hours:
{student_profile["Sleep_Hours"]}

Previous Scores:
{student_profile["Previous_Scores"]}

Tutoring Sessions:
{student_profile["Tutoring_Sessions"]}

Physical Activity:
{student_profile["Physical_Activity"]}

User question:

{user_question}
"""


    #starting conversation
    messages = [

        {
            "role": "system",
            "content": system_prompt
        },

        {
            "role": "user",
            "content": user_prompt
        }
    ]


    #agent loop
    for agent_step in range(5):


        #auto retry if server unavailable
        for attempt in range(3):

            try:

                response = client.chat.completions.create(

                    model="z-ai/glm-5.3-flash",

                    messages=messages,

                    tools=tools,

                    tool_choice="auto",

                    temperature=0.2
                )

                break


            except Exception as error:

                if attempt < 2:

                    time.sleep(2)

                    continue

                raise error


        #get model response
        message = response.choices[0].message


        #if agent does not request a tool,
        #return final answer
        if not message.tool_calls:

            if message.content:
                return message.content

            return (
                "The AI agent did not return "
                "a readable response."
            )


        #add agent's tool request
        #to conversation history
        messages.append(
            message
        )


        #run tools requested by agent
        for tool_call in message.tool_calls:

            tool_name = (
                tool_call.function.name
            )


            #read arguments generated by agent
            arguments = json.loads(
                tool_call.function.arguments
            )


            #execute statistical/support tool
            tool_result = execute_tool(
                tool_name,
                arguments
            )


            #send tool result back to agent
            messages.append({

                "role": "tool",

                "tool_call_id":
                    tool_call.id,

                "content":
                    json.dumps(
                        tool_result
                    )
            })


    #failsafe
    return (
        "The agent reached its maximum number "
        "of tool steps before producing a final response."
    )
