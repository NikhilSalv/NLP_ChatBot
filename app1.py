import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv('Student_course_data.csv')

# Drop unnecessary columns
df = df.drop(columns=['User_ID', 'Name', 'Courses Explored', 'Completion Status', 'Feedback Rating'])

# Preprocessing: One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=['Experience Level', 'Preferred Learning Mode', 'Area of Interest', 'Technologies of Interest'])

# Create a manual mapping for the course names to indices
course_mapping = {
    0: "Data Science Bootcamp",
    1: "Machine Learning with Python",
    2: "Web Development with Django",
    3: "Cloud Computing with AWS",
    4: "Full Stack JavaScript"
}

# Assuming 'Courses Enrolled' column has course names, map to indices manually
df_encoded['Courses Enrolled'] = df['Courses Enrolled'].map({v: k for k, v in course_mapping.items()})

# Define features (X) and target (y)
X = df_encoded.drop(columns=['Courses Enrolled'])
y = df_encoded['Courses Enrolled']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save the model
pickle.dump(clf, open("model.pkl", "wb"))

# Function to predict courses from a list of independent variables
def recommend_course(input_list, model, course_mapping):
    # Ensure input is in the right format for prediction (2D array)
    input_data = [input_list]
    
    # Predict course index
    predicted_course_index = model.predict(input_data)[0]
    
    # Map the predicted index to course name
    recommended_course = course_mapping[predicted_course_index]
    
    return recommended_course

# Example of using the model to predict for a new user
# Replace these values with actual values from the independent variables (same order as X columns)
new_user_data = [3.0, 0.0, 1.0, 0.0, 0.0, 1.0, 5.0]  # Example list of independent variables

# Load the model from the pickle file
model = pickle.load(open("model.pkl", "rb"))

# Get the recommended course for the new user
recommended_course = recommend_course(new_user_data, model, course_mapping)

print(f"Recommended course: {recommended_course}")
