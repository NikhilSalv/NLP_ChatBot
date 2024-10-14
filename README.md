## Course Recommendation System

## Overview

This project implements a Course Recommendation System using machine learning, integrated with a backend built on FastAPI. The system interacts with a Dialogflow-based chatbot to recommend the best courses based on user interactions and preferences. The machine learning model predicts the most suitable course for a user by analyzing various attributes like experience level, preferred learning mode, and areas of interest.

### Features

- Machine Learning Model: A RandomForestClassifier model trained to recommend courses based on user input.

- FastAPI Backend: The model is served using a FastAPI backend, which handles predictions via a RESTful API.
- Dialogflow Chatbot: A user-friendly chatbot deployed on Dialogflow, interacting with users and providing personalized course recommendations.
- Automated Model Deployment: The backend is designed to handle incoming requests and return recommendations dynamically.

  
## Technologies Used

- Machine Learning: scikit-learn, pandas
- API Framework: FastAPI
- Chatbot Platform: Dialogflow
- Model Persistence: pickle
- Deployment: Hosted using services like Heroku, AWS, or Google Cloud for scalable deployment.

## Dataset

The dataset used for this project contains information on:

- User experience level
- Preferred learning mode
- Areas of interest
- Technologies of interest
- Time commitment per week
- The dataset includes both categorical and numerical features, which are preprocessed using one-hot encoding.

### Installation

Requirements
Python 3.8 or higher
Required Python packages (listed in requirements.txt)
Install the dependencies:

#### bash
Copy code
pip install -r requirements.txt
Setting Up the Environment
Clone the repository:

### bash

Copy code

git clone https://github.com/yourusername/course-recommendation-system.git
cd course-recommendation-system

Download the dataset (if not already included):

Place the Student_course_data.csv file in the project root directory.
Train the model (if not using the pre-trained model):

bash

Copy code

python train_model.py

This will train the RandomForestClassifier and save the model as model.pkl.

Start the FastAPI server:

bash

Copy code

uvicorn app:app --reload

Dialogflow Integration:

Configure the Dialogflow agent to communicate with the FastAPI backend for course recommendations.
API Endpoints
GET /
Returns a welcome message.


POST /recommend_course
Takes user input in the form of a JSON object containing the required independent variables and returns a course recommendation.

Request Body:


json
Copy code
{
    "Time_commitment": 10,
    "Experience_Level_Advanced": 1,
    "Experience_Level_Beginner": 0,
    "Preferred_Learning_Mode_Instructor_led": 1,
    "Area_of_Interest_Front_End_Development": 1,
    "Technologies_of_Interest_Java_Python": 1
}

Response:

json
Copy code
{
    "recommended_course": "Data Science Bootcamp"
}
Project Structure
plaintext

Copy code
.
├── app.py                     # FastAPI application
├── train_model.py              # Script to train the ML model
├── model.pkl                   # Trained machine learning model
├── Student_course_data.csv      # Dataset
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation

Usage
Training the Model: To train the model, simply run the train_model.py script. It will load the dataset, preprocess it, and train a RandomForestClassifier. The trained model will be saved in the project as model.pkl.

bash

Copy code

python train_model.py

Running the FastAPI App: After the model is trained, start the FastAPI server to handle requests. The server will predict courses based on user input and return the result to the Dialogflow chatbot.

bash

Copy code

uvicorn app:app --reload

Dialogflow Chatbot: The chatbot interacts with users, collects necessary information, and calls the FastAPI API to recommend a course. Ensure the Dialogflow agent is configured to make an API call to the FastAPI server.

### How to Contribute :

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Commit your changes (git commit -am 'Add new feature').
Push the branch (git push origin feature-branch).
Open a pull request.
