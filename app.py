
from fastapi import FastAPI
import uvicorn
import pandas as pd
import joblib



app = FastAPI(debug=True)


@app.get("/")
def read_root():
    return {"Hello": "Universe"}

    
model_filename = "/Users/kushalsharma/Downloads/new_svm_with_grid_search.joblib"
loaded_pipeline = joblib.load(model_filename)

@app.get("/Predict")
def predict(Sex: str, Age: str, Height: str, Weight: str):

        # Convert Age to integer and Height/Weight to float
    try:
        Age = int(Age)
        Height = float(Height)
        Weight = float(Weight)
    except ValueError:
        return {"error": "Invalid input. Age must be an integer, Height and Weight must be numeric."}

    # Convert Sex to numeric code
    if Sex.lower() == "male":
        sex_code = 1
    elif Sex.lower() == "female":
        sex_code = 0
    else:
        return {"error": "Invalid input for Sex. Use 'male' or 'female'."}

    if Sex.lower() == "male":
        sex_code = 1
    else:
        sex_code = 0



    example_input = {
        "Sex": sex_code,
        "Age": Age,
        "Height": Height,
        "Weight": Weight,
    }


    input_df = pd.DataFrame([example_input])

    # Ensure column order matches the training data
    input_df = input_df[['Sex', 'Age', 'Height', 'Weight']]

    # Make prediction
    prediction = loaded_pipeline.predict(input_df)[0]

    # Return prediction
    return {"Detected as": prediction}

    
    

if  __name__ == "__main__":
    uvicorn.run(app, host= "0.0.0.0", port= 8000)















