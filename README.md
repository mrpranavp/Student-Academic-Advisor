# Student Academic Risk Advisor
<img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/4efb254e-c9da-4d52-999d-7241f636ede8" />

A multivariate statistical analysis and Agentic AI application for identifying students who may be academically at risk.
Developed as part of the **Multivariate Techniques** course.

## Application
[![Open the Application](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://student-risk-advisor.streamlit.app/)

Statistical Analysis- [https://colab.research.google.com/drive/1Ddk-lnvlTawVY6OZu3uidXIUKGEbkvfA?usp=sharing]

## Project Overview

Student academic performance depends on multiple factors including attendance, study hours, previous scores, sleep, tutoring and physical activity. This project analyses these factors using multivariate statistical techniques and develops an **LDA-based academic risk classifier**. The trained statistical model is integrated with an **Agentic AI system** that can call statistical tools, interpret predictions and provide academic support suggestions.

## Dataset

Name: Student Performance Factors

The dataset contains **6,607 students and 20 variables**.

Important numerical variables include:
- Hours Studied
- Attendance
- Sleep Hours
- Previous Scores
- Tutoring Sessions
- Physical Activity
- Exam Score

The analysis included data profiling, missing-value treatment, descriptive statistics, mean vector, covariance and correlation analysis and multivariate assumption checks.

## Multivariate Analysis

The project applies several concepts from Multivariate Techniques:

### Factor Analysis Suitability
Factor Analysis was considered but was not applied, as the diagnostics did not support a meaningful common-factor structure.
- KMO ≈ 0.498
- Bartlett's test p ≈ 0.060
This demonstrated the importance of checking whether a multivariate technique is appropriate before applying it.

### MANOVA
MANOVA was used to examine whether students with different motivation levels had different joint academic profiles.

**Pillai's Trace ≈ 0.0216, p < 0.001**
The overall difference was statistically significant but small in practical magnitude.

### LDA & QDA
Academic risk was operationally defined as:

**Exam Score ≤ 65**
Exam Score was excluded from the predictors to prevent target leakage.
| Model | Test Accuracy | ROC-AUC |
|------|------:|------:|
| LDA | 87.29% | 0.9455 |
| QDA | 86.38% | 0.9421 |

5-fold cross-validation produced an LDA ROC-AUC of approximately **0.9454**.
LDA was therefore selected as the final classifier.

The strongest discriminating variables were:
1. Attendance
2. Hours Studied
3. Previous Scores

### Canonical Correlation Analysis

CCA was used to examine the relationship between:

**Engagement:** Hours Studied + Attendance  
**Performance:** Previous Scores + Exam Score
The first canonical correlation was approximately **0.747**, indicating one dominant multivariate relationship between the two variable sets.

## Agentic AI Application
The statistical model was integrated into a tool-calling AI agent.

The agent follows:
**Perceive → Reason → Act → Observe → Respond**

Architecture:
Student Profile -> Streamlit Application -> OpenRouter AI Agent -> Statistical Tool Calling -> LDA Risk Prediction / Driver Analysis / Academic Support -> Agent Interpretation

*Note:* The AI does **not generate the risk probability itself**. Risk classification and probability are calculated by the trained LDA model, while the AI agent decides which tools to call and explains their results.

## Technologies
- Python
- Google Colab
- pandas / NumPy
- scikit-learn
- statsmodels
- Streamlit
- OpenRouter
- GLM 5.3 Flash
- GitHub

## Limitations

- The dataset is observational and results do not establish causation.
- The academic-risk threshold is project-specific and is not an official failure criterion.
- Some multivariate assumptions were not fully satisfied and are documented in the analysis.
- Academic recommendations are decision-support suggestions rather than validated interventions.

## AI Assistance Disclosure

The statistical analysis, methodological decisions, interpretation, model selection, validation and overall project design were carried out by the student. Generative AI was used primarily to assist with coding, debugging, code refinement and documentation. AI-assisted code was reviewed, tested and adapted before inclusion in this project.

