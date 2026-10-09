# Student Performance Prediction System
## Comprehensive Technical Project Report

---

### 1. Title Page
* **Project Title:** Student Performance Prediction System
* **Author:** Aryan Mayur Mange
* **Role:** Machine Learning Intern
* **Target Metric:** Continuous Final Examination Score (`Exam_Score` $\in [0, 100]$)
* **Status:** Complete & Submission-Ready

---

### 2. Introduction
In educational institutions, identifying students who may need additional academic support often happens too late in the academic cycle. This project implements a machine learning regression system to predict student performance early using behavioral habits, school engagement, and environmental attributes. By anticipating student outcomes, educators can provide timely, targeted interventions.

---

### 3. Objective
* Model final examination performance as a supervised regression task on a continuous scale $[0, 100]$.
* Identify the key drivers of student achievement across behavioral and lifestyle factors.
* Benchmark standard linear models against non-linear tree-based ensembles.
* Deploy an end-to-end inference pipeline capable of taking raw student inputs and generating predictions.

---

### 4. Dataset Overview
* **Source:** Kaggle Student Performance Factors Dataset.
* **Volume:** 6,607 student records.
* **Attributes:** 20 total variables (6 numerical features, 13 categorical features, 1 regression target).
* **Target Feature:** `Exam_Score` (mean = 67.24, std = 3.89, continuous range 0–100).
* **Predictor Features:**
  - *Numerical:* `Hours_Studied`, `Attendance`, `Sleep_Hours`, `Previous_Scores`, `Tutoring_Sessions`, `Physical_Activity`.
  - *Categorical:* `Parental_Involvement`, `Access_to_Resources`, `Extracurricular_Activities`, `Motivation_Level`, `Internet_Access`, `Family_Income`, `Teacher_Quality`, `School_Type`, `Peer_Influence`, `Learning_Disabilities`, `Parental_Education_Level`, `Distance_from_Home`, `Gender`.

---

### 5. Data Preprocessing & Cleaning
* **Missing Value Imputation:** Categorical columns (`Teacher_Quality`, `Parental_Education_Level`, `Distance_from_Home`) were imputed using mode replacement.
* **Boundary Anomaly Capping:** Any out-of-range exam scores ($> 100$) were strictly capped at 100.
* **Deduplication:** Audited the full dataset, confirming 0 duplicate rows across all 6,607 samples.
* **Transformations:** Standardized continuous predictors using `StandardScaler` ($\mu = 0, \sigma = 1$) and encoded categorical features with `OneHotEncoder(drop='first')`, producing a 27-dimensional feature space.

---

### 6. Exploratory Data Analysis (EDA)
* `Exam_Score` follows an approximately normal distribution centered around 67.2 marks.
* Attendance and study hours show strong positive linear trends with final exam marks.
* Lifestyle factors like sleep hours ($r = -0.02$) and physical activity ($r = 0.03$) showed minimal direct linear influence on performance.

---

### 7. Feature Selection Justification
* **Predictive Value:** `Attendance` ($r = 0.58$) and `Hours_Studied` ($r = 0.45$) serve as the primary numerical signals.
* **Categorical Drivers:** Categorical features such as `Parental_Involvement` and `Access_to_Resources` showed statistically significant differences across groups via ANOVA ($p < 0.001$).
* **Multicollinearity:** Pairwise correlations between features remained below $\vert{}r\vert{} < 0.05$, confirming an absence of multicollinearity.

---

### 8. Model Building & Training
* **Splitting:** 80% Training ($n = 5,285$) and 20% Testing ($n = 1,322$) with a fixed seed (`random_state=42`).
* **Algorithms Tested:**
  1. Ordinary Least Squares (OLS) Linear Regression
  2. Ridge Regression ($L_2$ regularization, $\alpha = 1.0$)
  3. Lasso Regression ($L_1$ regularization, $\alpha = 0.01$)
  4. Decision Tree Regressor (`max_depth=6`)
  5. Random Forest Regressor (`n_estimators=100`, `max_depth=8`)

---

### 9. Model Evaluation Metrics
* **Mean Absolute Error (MAE):** 1.712 marks (average prediction error on a 100-mark scale).
* **Root Mean Squared Error (RMSE):** 2.148 marks (similar to MAE, confirming the absence of large outlier errors).
* **Coefficient of Determination ($R^2$):** 0.695 (the model accounts for ~69.5% of exam score variance).
* **Generalization Check:** Training $R^2$ (0.697) and Testing $R^2$ (0.695) indicate balanced fit without overfitting.

---

### 10. Model Comparison
* Linear Regression and Random Forest showed comparable test performance ($R^2 \approx 0.695$ vs. $0.696$).
* Linear Regression was selected as the operational baseline due to its faster inference time, direct interpretability, and stable coefficients.

---

### 11. Prediction System Implementation
* Serialized the complete preprocessing pipeline into `model/preprocessor.joblib` and the model into `model/best_student_model.joblib`.
* Implemented `src/predict.py` to ingest raw student feature dictionaries, apply the saved transformations, and return bounded predictions $[0, 100]$.

---

### 12. Results on Sample Student Profiles
* **High Engagement (Student A):** 96% attendance, 28 study hrs/wk $\rightarrow$ **74.12 marks**
* **Average Profile (Student B):** 78% attendance, 17 study hrs/wk $\rightarrow$ **67.05 marks**
* **At-Risk Profile (Student C):** 52% attendance, 5 study hrs/wk $\rightarrow$ **58.94 marks**

---

### 13. Conclusion
Academic performance in this dataset is primarily driven by effort indicators (class attendance and independent study hours), supported by home resources and parental engagement. Linear regression provides an accurate and explainable model for this dataset, maintaining an average error margin of ~1.7 marks.

---

### 14. Future Scope
* **Interactive UI:** Build a Streamlit dashboard for non-technical school staff.
* **Longitudinal Data:** Track multi-semester trends rather than single exam snapshots.
* **Automated Early Warnings:** Trigger automated alerts when predicted marks drop below passing thresholds ($< 50$ marks).