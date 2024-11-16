import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mangum import Mangum  # AWS Lambda integration with FastAPI
import boto3
import os
# App name
app = FastAPI(title="Credit Card Risk Modelling for Loan")

# AWS S3 bucket and file information
BUCKET_NAME = "loan-prediction-model-11-15-2024"
MODEL_FILE_NAME = "RandomForest_Best.sav"

# Initialize model variable
RF_pipeline = None

# Load the machine learning pipeline on startup

@app.on_event("startup")
def load_ml_pipeline():
    """
    Load the machine learning pipeline from an S3 bucket.
    """
    global RF_pipeline
    try:
        # Initialize S3 client
        s3 = boto3.client("s3")

        # File path for temporary storage in Lambda
        temp_file_path = f"{MODEL_FILE_NAME}"

        # Check if the file exists already to avoid re-downloading
        if not os.path.exists(temp_file_path):
            # Download the model from S3
            s3.download_file(BUCKET_NAME, MODEL_FILE_NAME, temp_file_path)
            print(f"Model downloaded from S3: {temp_file_path}")
        
        # Load the model using joblib
        RF_pipeline = joblib.load(temp_file_path)
        print("Model loaded successfully from /tmp/")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Failed to load the model.")

# Define the data model for incoming prediction requests
class Credit_Model(BaseModel):
    person_age: int
    person_income: int
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: int
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int

# API endpoint for making predictions
@app.post("/predict")
def lambda_handler(inference_request: Credit_Model):
    try:
        # Convert incoming data into a DataFrame
        input_dictionary = {
            "person_age": inference_request.person_age,
            "person_income": inference_request.person_income,
            "person_home_ownership": inference_request.person_home_ownership,
            "person_emp_length": inference_request.person_emp_length,
            "loan_intent": inference_request.loan_intent,
            "loan_grade": inference_request.loan_grade,
            "loan_amnt": inference_request.loan_amnt,
            "loan_int_rate": inference_request.loan_int_rate,
            "loan_percent_income": inference_request.loan_percent_income,
            "cb_person_default_on_file": inference_request.cb_person_default_on_file,
            "cb_person_cred_hist_length": inference_request.cb_person_cred_hist_length
        }

        # Create a DataFrame for model prediction
        inference_request_Data = pd.DataFrame(input_dictionary, index=[0])

        # Predict using the loaded model
        prediction = RF_pipeline.predict(inference_request_Data)
        result = "Not Defaulted" if prediction == 0 else "Defaulted"

        return {"Prediction": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Lambda handler using Mangum for AWS Lambda integration
handler = Mangum(app)
