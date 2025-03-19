import joblib
import pandas as pd
import numpy as np
from pickle import load

model = joblib.load("model_abid.pkl")

# Example: Test data
new_data = pd.DataFrame({
    "Gender": [1],
    "Age": [60],
    "Urea": [5.7],
    "Creatine":[70],
    "HbA1c": [12.7],
    "Cholesterol": [5],
    "Trigliserida": [2.5],
    "VLDL": [0.7],
    "BMI": [36]
})

# Preprocess new_data (e.g., scaling and encoding) before prediction, as discussed earlier
predictions = model.predict(new_data)

# Map predictions back to original labels
label_mapping = {0: "Non Diabetes", 1: "Pra-diabetes", 2: "Diabetes"}
predicted_classes = [label_mapping[pred] for pred in predictions]

print("Predicted Classes:", predicted_classes)