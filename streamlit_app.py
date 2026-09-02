import streamlit as st
from agent import run_student_agent


st.set_page_config(
    page_title="Student Academic Risk Advisor",
    page_icon="🎓",
    layout="wide"
)


st.title("Student Academic Risk Advisor")

st.write(
    """
    An Agentic AI decision-support system combining
    **Linear Discriminant Analysis (LDA)** with a
    **Gemini AI agent**.

    The AI agent can call the trained statistical model,
    analyse important profile characteristics and provide
    academic-support recommendations.
    """
)

st.divider()


#profile
st.subheader("Student Profile")

col1, col2 = st.columns(2)


with col1:

    hours_studied = st.number_input(
        "Hours Studied",
        min_value=0,
        max_value=50,
        value=20
    )

    attendance = st.slider(
        "Attendance (%)",
        0,
        100,
        80
    )

    sleep_hours = st.slider(
        "Sleep Hours",
        0.0,
        12.0,
        7.0,
        0.5
    )


with col2:

    previous_scores = st.slider(
        "Previous Score",
        0,
        100,
        75
    )

    tutoring_sessions = st.number_input(
        "Tutoring Sessions",
        min_value=0,
        max_value=10,
        value=1
    )

    physical_activity = st.number_input(
        "Physical Activity",
        min_value=0,
        max_value=10,
        value=3
    )


student_profile = {

    "Hours_Studied": hours_studied,
    "Attendance": attendance,
    "Sleep_Hours": sleep_hours,
    "Previous_Scores": previous_scores,
    "Tutoring_Sessions": tutoring_sessions,
    "Physical_Activity": physical_activity

}


st.divider()

#asking agentic ai
st.subheader("Ask the AI Academic Advisor")

user_question = st.text_area(
    "What would you like the agent to analyse?",
    value=(
        "Assess this student's academic risk, explain "
        "the most important characteristics, and suggest "
        "what the student could improve."
    ),
    height=100
)


if st.button(
    "Ask the Agent",
    type="primary",
    use_container_width=True
):

    if not user_question.strip():

        st.warning(
            "Please enter a question for the agent."
        )

    else:

        with st.spinner(
            "Agent AI is analysing the student's profile..."
        ):

            try:

                answer = run_student_agent(
                    student_profile,
                    user_question
                )

                st.subheader("AI Agent Response")

                st.markdown(answer)

            except Exception as error:

                st.error(
                    f"Agent error: {error}"
                )


st.divider()


with st.expander(
    "How does the Agentic AI system work?"
):

    st.markdown(
        """
        **1. Perceive**  
        The Gemini agent receives the student's profile
        and the user's question.

        **2. Reason**  
        The agent determines which statistical tools are
        required.

        **3. Act**  
        Gemini can call:

        - `risk_prediction_tool` — trained LDA classifier
        - `risk_driver_tool` — LDA contribution analysis
        - `academic_support_tool` — support recommendations

        **4. Observe**  
        The tool results are returned to Gemini.

        **5. Respond**  
        The agent interprets the statistical results and
        produces a student-friendly response.
        """
    )


st.caption(
    """
    Academic demonstration only. Risk estimates represent
    statistical patterns in the project dataset and are not
    official institutional classifications or causal claims.
    """
)
