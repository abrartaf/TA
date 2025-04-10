from flask import Flask, request, render_template, session, redirect, url_for
import pickle
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
import os

app = Flask(__name__)
app.secret_key = 'Sisfor21'

# Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/ta_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database model
class ta(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    Gender = db.Column(db.String(10), nullable=False)
    Age = db.Column(db.Integer, nullable=False)
    Urea = db.Column(db.Float)
    Creatine = db.Column(db.Float)
    HbA1c = db.Column(db.Float, nullable=False)
    Cholesterol = db.Column(db.Float, nullable=False)
    Triglycerides = db.Column(db.Float, nullable=False)
    VLDL = db.Column(db.Float)
    BMI = db.Column(db.Float, nullable=False)
    Prediction = db.Column(db.String(50), nullable=False)

# Load models
with open('rf.pkl', 'rb') as f:
    model_farhan = pickle.load(f)
with open('model_abid.pkl', 'rb') as f:
    model_abid = pickle.load(f)

label_mapping = {0: "Non Diabetes", 1: "Pre-diabetes", 2: "Diabetes"}

@app.route("/")
def home():
    session.clear()  # Reset semua hasil jika kembali ke home
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction_abid = session.get("prediction_abid")
    prediction_farhan = session.get("prediction_farhan")
    form_data = session.get("form_data", {})

    if request.method == "POST":
        selected_model = request.form.get("model")

        # Ambil data dari form dan simpan di session
        form_data = {
            "Gender": request.form["Gender"],
            "Age": request.form["Age"],
            "Urea": request.form["Urea"],
            "Creatine": request.form["Creatine"],
            "HbA1c": request.form["HbA1c"],
            "Cholesterol": request.form["Cholesterol"],
            "Trigliserida": request.form["Trigliserida"],
            "VLDL": request.form["VLDL"],
            "BMI": request.form["BMI"]
        }
        session["form_data"] = form_data

        try:
            Gender_encoded = 1 if form_data["Gender"].lower() == "male" else 0
            input_data = pd.DataFrame([[
                Gender_encoded,
                int(form_data["Age"]),
                float(form_data["Urea"]),
                float(form_data["Creatine"]),
                float(form_data["HbA1c"]),
                float(form_data["Cholesterol"]),
                float(form_data["Trigliserida"]),
                float(form_data["VLDL"]),
                float(form_data["BMI"])
            ]], columns=['Gender', 'Age', 'Urea', 'Creatine', 'HbA1c',
                         'Cholesterol', 'Trigliserida', 'VLDL', 'BMI'])

            # Prediksi berdasarkan tombol
            if selected_model == "abid":
                result = model_abid.predict(input_data)[0]
                prediction_abid = label_mapping.get(result, "Unknown")
                session["prediction_abid"] = prediction_abid

            elif selected_model == "farhan":
                result = model_farhan.predict(input_data)[0]
                prediction_farhan = label_mapping.get(result, "Unknown")
                session["prediction_farhan"] = prediction_farhan

        except Exception as e:
            if selected_model == "abid":
                session["prediction_abid"] = f"Error: {str(e)}"
            elif selected_model == "farhan":
                session["prediction_farhan"] = f"Error: {str(e)}"

        return redirect(url_for("predict"))

    return render_template("model_both.html",
                           prediction_abid=prediction_abid,
                           prediction_farhan=prediction_farhan,
                           form_data=form_data)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
