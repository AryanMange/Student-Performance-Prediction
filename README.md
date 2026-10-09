\# Student Performance Prediction System 🎓📊



An end-to-end Machine Learning regression project that predicts student final exam scores (`Exam\_Score`, 0–100) based on academic habits, socioeconomic factors, and institutional support.



\---



\## 📌 Project Overview

Identifying at-risk students before final exams allows educators to step in early. This project builds a regression pipeline using 6,607 student records to forecast exam outcomes from 19 behavioral and demographic variables.



\* \*\*Target Variable:\*\* `Exam\_Score` (0–100 continuous marks)

\* \*\*Dataset Size:\*\* 6,607 student records, 20 attributes

\* \*\*Selected Model:\*\* Linear Regression / Random Forest Regressor

\* \*\*Core Drivers:\*\* Attendance ($r = 0.58$) and Study Hours ($r = 0.45$)



\---



\## 🏗️ Repository Structure



```text

Student-Performance-Prediction/

├── data/

│   ├── student\_data.csv                    # Raw dataset

│   ├── student\_data\_cleaned.csv            # Cleaned dataset

│   ├── X\_train.csv                         # Training features (5,285 x 27)

│   ├── X\_test.csv                          # Testing features (1,322 x 27)

│   ├── y\_train.csv                         # Training targets

│   └── y\_test.csv                          # Testing targets

├── notebook/

│   ├── 01\_day1\_to\_05\_eda\_and\_cleaning.ipynb

│   ├── 06\_day6\_feature\_engineering.ipynb

│   ├── 07\_day7\_week1\_review.ipynb

│   ├── 08\_day8\_train\_test\_split\_and\_baseline.ipynb

│   ├── 09\_day9\_model\_evaluation.ipynb

│   ├── 10\_day10\_model\_comparison.ipynb

│   ├── 11\_day11\_build\_prediction\_system.ipynb

│   └── 12\_day12\_system\_testing\_and\_visualizations.ipynb

├── visualizations/

│   ├── 01\_exam\_score\_distribution.png

│   ├── 02\_attendance\_vs\_exam\_score.png

│   ├── 03\_hours\_studied\_vs\_exam\_score.png

│   ├── 04\_previous\_scores\_vs\_exam\_score.png

│   ├── 05\_environmental\_factors\_boxplot.png

│   ├── 06\_correlation\_heatmap.png

│   ├── 07\_baseline\_residual\_analysis.png

│   ├── 08\_model\_comparison\_chart.png

│   ├── 09\_actual\_vs\_predicted.png

│   └── 10\_feature\_importance.png

├── model/

│   ├── best\_student\_model.joblib           # Trained model artifact

│   └── preprocessor.joblib                 # ColumnTransformer artifact

├── src/

│   └── predict.py                          # Programmatic inference module

├── docs/

│   └── PROJECT\_REPORT.md                   # 14-section formal report

├── requirements.txt

└── README.md







⚙️ Machine Learning Pipeline\[Raw Dataset]

&#x20;     ↓ (Mode Imputation \& 100-Mark Cap)

\[Cleaned Data]

&#x20;     ↓ (StandardScaler for numerics + OneHotEncoder for categoricals)

\[80/20 Train-Test Split] (5,285 train / 1,322 test)

&#x20;     ↓

\[Model Benchmarking] (Linear Regression, Ridge, Lasso, Decision Tree, Random Forest)

&#x20;     ↓

\[Model Evaluation] (MAE: \~1.71 marks, RMSE: \~2.15 marks, R²: \~0.695)

&#x20;     ↓

\[Inference Pipeline] (src/predict.py -> Bounded predictions \[0, 100])





📈 Model Performance Benchmark



Models evaluated on the unseen 20% test split ($n = 1,322$):





🚀 How to Run the Prediction System1.



Installation



git clone \[https://github.com/aryanmange/Student-Performance-Prediction.git](https://github.com/aryanmange/Student-Performance-Prediction.git)

cd Student-Performance-Prediction

pip install -r requirements.txt





2\. Run Inference



PowerShell



python src/predict.py





3\. Programmatic Usage in Python

from src.predict import predict\_score



sample\_student = {

&#x20;   'Hours\_Studied': 20,

&#x20;   'Attendance': 85,

&#x20;   'Sleep\_Hours': 7,

&#x20;   'Previous\_Scores': 75,

&#x20;   'Tutoring\_Sessions': 1,

&#x20;   'Physical\_Activity': 3,

&#x20;   'Parental\_Involvement': 'High',

&#x20;   'Access\_to\_Resources': 'High',

&#x20;   'Extracurricular\_Activities': 'Yes',

&#x20;   'Motivation\_Level': 'Medium',

&#x20;   'Internet\_Access': 'Yes',

&#x20;   'Family\_Income': 'Medium',

&#x20;   'Teacher\_Quality': 'Medium',

&#x20;   'School\_Type': 'Public',

&#x20;   'Peer\_Influence': 'Neutral',

&#x20;   'Learning\_Disabilities': 'No',

&#x20;   'Parental\_Education\_Level': 'College',

&#x20;   'Distance\_from\_Home': 'Near',

&#x20;   'Gender': 'Female'

}



score = predict\_score(sample\_student)

print(f"Predicted Student Performance: {score} marks")





🔮 Future Scope

Web Interface: Deploy a Streamlit or FastAPI app for interactive use.

Longitudinal Tracking: Include semester-by-semester historical grades.

Early Warning Alerts: Automatic notifications for predicted scores below passing thresholds.

