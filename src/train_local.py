import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

# 1. Load data
candidates = [
    os.path.join("data", "student_data_cleaned.csv"),
    "student_data_cleaned.csv",
    os.path.join("data", "student_data.csv"),
    "student_data.csv"
]
data_path = next((p for p in candidates if os.path.exists(p)), None)
if not data_path:
    raise FileNotFoundError("Could not find dataset in project folder or data/!")

df = pd.read_csv(data_path)

# Ensure mode imputation and bounds capping
for col in ['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home']:
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])
df.loc[df['Exam_Score'] > 100, 'Exam_Score'] = 100

X = df.drop(columns=['Exam_Score'])
y = df['Exam_Score']

num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()

# 2. Build preprocessor & train model
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
    ],
    remainder='drop'
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train_processed = preprocessor.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_processed, y_train)

# 3. Save matching artifacts
os.makedirs("model", exist_ok=True)
joblib.dump(preprocessor, os.path.join("model", "preprocessor.joblib"))
joblib.dump(model, os.path.join("model", "best_student_model.joblib"))

print("Successfully generated local artifacts compatible with your scikit-learn version!")