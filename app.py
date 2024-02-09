import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI
import joblib
import pandas as pd

# Create the app object
app = FastAPI()

# Define the input data model using Pydantic
class InputData(BaseModel):
    Sex: int
    Age: int
    Height: int
    Weight: int

# Define a dictionary mapping numerical labels to string labels
label_mapping = {
    "0": "Overweight",
    "1": "Stunting",
    "2": "Underweight",
    "3": "Wasting"
}

# Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# Prediction route, accepts POST requests with JSON data
@app.post('/predict')
def predict(input_data: InputData):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data.model_dump()])

    # Load the trained SVM model using joblib
    model_filename = '/Users/kushalsharma/Desktop/MalnuDetect/svm_model.joblib'
    loaded_pipeline = joblib.load(model_filename)

    # Make predictions using the loaded pipeline
    prediction = loaded_pipeline.predict(input_df)[0]

    # Map numerical prediction to string label
    prediction_str = label_mapping.get(str(prediction), "Unknown")

    return {"prediction": prediction_str}

# Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
