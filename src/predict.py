import os
import joblib
import pandas as pd
import numpy as np

def get_artifacts():
    # Resolve project root dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    
    # Check current directory and project root for model folder
    candidate_dirs = [
        os.path.join(project_root, "model"),
        os.path.join(current_dir, "model"),
        "model"
    ]
    
    m_path, p_path = None, None
    for d in candidate_dirs:
        m_test = os.path.join(d, "best_student_model.joblib")
        p_test = os.path.join(d, "preprocessor.joblib")
        if os.path.exists(m_test) and os.path.exists(p_test):
            m_path = m_test
            p_path = p_test
            break
            
    if not m_path:
        raise FileNotFoundError("Could not find 'best_student_model.joblib' and 'preprocessor.joblib' in 'model/' folder.")
        
    model = joblib.load(m_path)
    prep = joblib.load(p_path)
    return model, prep

def predict_score(student_features: dict) -> float:
    """
    Accepts raw student features dictionary and returns predicted Exam_Score bound [0, 100].
    """
    model, prep = get_artifacts()
    df_input = pd.DataFrame([student_features])
    features_transformed = prep.transform(df_input)
    prediction = model.predict(features_transformed)[0]
    return round(float(np.clip(prediction, 0, 100)), 2)

if __name__ == "__main__":
    sample_student = {
        'Hours_Studied': 20,
        'Attendance': 85,
        'Sleep_Hours': 7,
        'Previous_Scores': 75,
        'Tutoring_Sessions': 1,
        'Physical_Activity': 3,
        'Parental_Involvement': 'High',
        'Access_to_Resources': 'High',
        'Extracurricular_Activities': 'Yes',
        'Motivation_Level': 'Medium',
        'Internet_Access': 'Yes',
        'Family_Income': 'Medium',
        'Teacher_Quality': 'Medium',
        'School_Type': 'Public',
        'Peer_Influence': 'Neutral',
        'Learning_Disabilities': 'No',
        'Parental_Education_Level': 'College',
        'Distance_from_Home': 'Near',
        'Gender': 'Female'
    }
    result = predict_score(sample_student)
    print(f"Sample Student Predicted Score: {result} / 100")
