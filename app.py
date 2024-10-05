from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI()

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Define a mapping between the course indices and course names
course_mapping = {
    0: "Data Science Bootcamp",
    1: "Machine Learning with Python",
    2: "Web Development with Django",
    3: "Cloud Computing with AWS",
    4: "Full Stack JavaScript",
    # Add other courses as needed
}

# Define a request body model with the necessary features
class UserData(BaseModel):
    Experience_Level: str
    Preferred_Learning_Mode: str
    Area_of_Interest: str
    Technologies_of_Interest: str
    Feature_1: float  # Replace with actual feature names as per your dataset
    Feature_2: float  # Replace with actual feature names as per your dataset
    Feature_3: float  # Replace with actual feature names as per your dataset
    # Add other numerical features as needed

# FastAPI route to predict top 2 recommended courses
@app.post("/recommend_courses/")
def recommend_courses(user_data: UserData):
    # Convert user input to a DataFrame
    user_df = pd.DataFrame([user_data.dict()])
    
    # One-hot encode the categorical fields to match the model's training data
    categorical_columns = ['Experience_Level', 'Preferred_Learning_Mode', 'Area_of_Interest', 'Technologies_of_Interest']
    user_df_encoded = pd.get_dummies(user_df, columns=categorical_columns)

    # Re-align the columns to match the training set
    model_columns = pickle.load(open('model.pkl', 'rb'))  # Columns from training
    user_df_encoded = user_df_encoded.reindex(columns=model_columns, fill_value=0)

    # Predict course probabilities for the user
    course_probs = model.predict_proba(user_df_encoded)[0]
    
    # Get the indices of the top recommended courses
    top_courses_indices = course_probs.argsort()[-2:][::-1]
    
    # Map the indices to course names using the manual course mapping
    recommended_courses = [course_mapping[idx] for idx in top_courses_indices]
    
    return {"recommended_courses": recommended_courses}

# Save model columns for re-aligning the user input DataFrame
if __name__ == '__main__':
    import os
    if not os.path.exists('model_columns.pkl'):
        pickle.dump(list(X.columns), open('model_columns.pkl', 'wb'))
