from locust import HttpUser, task, between

class DiabetesPredictionUser(HttpUser):
    wait_time = between(1, 3)  # waktu tunggu antar request

    @task
    def predict_diabetes(self):
        self.client.post("/predict", data={
            "Gender": "Male",
            "Age": "49",
            "Urea": "3.1",
            "Creatine": "75",
            "HbA1c": "6",
            "Cholesterol": "3.6",
            "Trigliserida": "2.4",
            "VLDL": "0.6",
            "BMI": "24"
        })
