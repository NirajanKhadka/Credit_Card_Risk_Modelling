import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Request
import json

# App name
app = FastAPI(title="Credit Card Risk Modelling for Loan")

# Setting the path for static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
def load_ml_pipeline():
    # Load the machine learning pipeline from .sav format
    global RF_pipeline
    RF_pipeline = joblib.load(open('ML_artifact/RandomForest_Best.sav', 'rb'))

# Defining base class to represent data points for predictions
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

# Route to render the home page with the form
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        data = json.loads(contents)
        inference_request = Credit_Model(**data)  # Use the model to validate and parse the data
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format"}
    except Exception as e:
        return {"error": str(e)}

    # Process prediction
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

    inference_request_Data = pd.DataFrame(input_dictionary, index=[0])
    prediction = RF_pipeline.predict(inference_request_Data)
    result = "Not Defaulted" if prediction == 0 else "Defaulted"

    return {"Prediction": result}



# REST API for prediction (Returns JSON)
@app.post("/predict")
def predict(inference_request: Credit_Model):
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

    inference_request_Data = pd.DataFrame(input_dictionary, index=[0])
    prediction = RF_pipeline.predict(inference_request_Data)
    result = "Not Defaulted" if prediction == 0 else "Defaulted"

    return {"Prediction": result}