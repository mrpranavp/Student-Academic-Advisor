import streamlit as st
import pandas as pd

from agent import student_advisory_agent


st.set_page_config(
    page_title="Student Academic Risk Advisor",
    page_icon="🎓",
    layout="wide"
)


st.title("Student Academic Risk Advisor")

st.write(
    """
    This agentic decision-support application uses
    **Linear Discriminant Analysis (LDA)** to estimate
    whether a student belongs to the lower-performing
    segment of the study dataset.

    The agent combines statistical risk prediction,
    model interpretation, and academic-support
    recommendations.
    """
)

st.divider()

#input
st.subheader("Student Profile")

col1, col2 = st.columns(2)


with col1:

    hours_studied = st.number_input(
        "Hours Studied",
        min_value=0,
        max_value=50,
        value=20,
        step=1
    )

    attendance = st.slider(
        "Attendance (%)",
        min_value=0,
        max_value=100,
        value=80
    )

    sleep_hours = st.slider(
        "Sleep Hours",
        min_value=0.0,
        max_value=12.0,
        value=7.0,
        step=0.5
    )


with col2:

    previous_scores = st.slider(
        "Previous Score",
        min_value=0,
        max_value=100,
        value=75
    )

    tutoring_sessions = st.number_input(
        "Tutoring Sessions",
        min_value=0,
        max_value=10,
        value=1,
        step=1
    )

    physical_activity = st.number_input(
        "Physical Activity",
        min_value=0,
        max_value=10,
        value=3,
        step=1
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

#analysis
if st.button(
    "Analyse Student",
    type="primary",
    use_container_width=True
):

    result = student_advisory_agent(
        student_profile
    )

    risk_result = result["risk"]

    probability = (
        risk_result["risk_probability"] * 100
    )
# risk checking
    st.subheader("Risk Assessment")

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            label="Estimated Risk Probability",
            value=f"{probability:.1f}%"
        )


    with col2:

        if risk_result["classification"] == "At Risk":

            st.error(
                "AT RISK! ❌"
            )

        else:

            st.success(
                "NOT AT RISK! ✅"
            )


    st.progress(
        min(
            risk_result["risk_probability"],
            1.0
        )
    )

#agent interpretation 
    st.subheader(
        "Most Influential Profile Characteristics"
    )

    st.write(
        """
        These characteristics had the largest
        contributions to the LDA classification for
        this student.
        """
    )

    drivers = result["drivers"].copy()

    drivers["Variable"] = (
        drivers["Variable"]
        .str.replace("_", " ")
    )

    display_drivers = drivers[
        [
            "Variable",
            "Student_Value"
        ]
    ].rename(
        columns={
            "Variable": "Characteristic",
            "Student_Value": "Student Value"
        }
    )

    st.dataframe(
        display_drivers,
        use_container_width=True,
        hide_index=True
    )


#charts
    chart_data = drivers[
        [
            "Variable",
            "Absolute_Contribution"
        ]
    ].copy()

    chart_data = chart_data.set_index(
        "Variable"
    )

    st.bar_chart(
        chart_data
    )

#recomendation
    st.subheader("Academic Advisory")

    for recommendation in result[
        "recommendations"
    ]:

        st.write(
            f"• {recommendation}"
        )

# workflow
    with st.expander(
        "How did the agent make this assessment?"
    ):

        st.write(
            """
            **1. Perceive:**  
            The agent reads the student's academic profile.

            **2. Statistical tool call:**  
            The profile is standardized using the scaler
            fitted on the training data and passed to the
            trained Linear Discriminant Analysis model.

            **3. Observe:**  
            The agent receives the predicted class and
            estimated risk probability.

            **4. Interpret:**  
            LDA contributions are examined to identify
            characteristics that played the largest role
            in the student's classification.

            **5. Act:**  
            The agent generates academic-support
            recommendations based on the student's
            observed profile.
            """
        )


    st.info(
        """
        The 'At Risk' category was defined in this project
        using the lower-performing segment of the study
        dataset. It is a statistical decision-support
        estimate and is not an official institutional
        pass/fail classification.

        Model relationships should not be interpreted as
        evidence that these factors causally determine
        academic performance.
        """
    )
