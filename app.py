from flask import Flask, request, render_template
import pickle
import pandas as pd
from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__)


app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/ta_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy()
db.init_app(app)

# Define the Database Model
class ta(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    Gender = db.Column(db.String(10), nullable=False)
    Age = db.Column(db.Integer, nullable=False)
    Urea = db.Column(db.Float, nullable=True)
    Creatine = db.Column(db.Float, nullable=True)
    HbA1c = db.Column(db.Float, nullable=False)
    Cholesterol = db.Column(db.Float, nullable=False)
    Triglycerides = db.Column(db.Float, nullable=False)
    VLDL = db.Column(db.Float, nullable=True)
    BMI = db.Column(db.Float, nullable=False)
    Prediction = db.Column(db.String(50), nullable=False)

# Load Machine Learning Models
with open('rf.pkl', 'rb') as model_file_farhan:
    model_farhan = pickle.load(model_file_farhan)

with open('model_abid.pkl', 'rb') as model_file_abid:
    model_abid = pickle.load(model_file_abid)

# Label Mapping for Predictions
label_mapping = {0: "Non Diabetes", 1: "Pre-diabetes", 2: "Diabetes"}

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def prediction_both():
    return render_template("model_both.html")

@app.route("/predictBoth", methods=['POST'])
def predict_both_models():
    label_mapping = {0: "Non Diabetes", 1: "Pre-diabetes", 2: "Diabetes"}
    prediction_abid = None
    prediction_farhan = None

    try:
        # Ambil data dari form
        Gender = request.form['Gender']
        Age = int(request.form['Age'])
        Urea = float(request.form['Urea'])
        Creatine = float(request.form['Creatine'])
        HbA1c = float(request.form['HbA1c'])
        Cholesterol = float(request.form['Cholesterol'])
        Triglycerides = float(request.form['Trigliserida'])
        VLDL = float(request.form['VLDL'])
        BMI = float(request.form['BMI'])

        # Encode gender
        Gender_encoded = 1 if Gender.lower() == 'male' else 0

        # Buat satu DataFrame untuk kedua model
        input_data = pd.DataFrame([[
            Gender_encoded, Age, Urea, Creatine, HbA1c,
            Cholesterol, Triglycerides, VLDL, BMI
        ]], columns=[
            'Gender', 'Age', 'Urea', 'Creatine', 'HbA1c',
            'Cholesterol', 'Trigliserida', 'VLDL', 'BMI'
        ])

        # Prediksi dengan model Abid
        result_abid = model_abid.predict(input_data)[0]
        prediction_abid = label_mapping.get(result_abid, "Unknown")

        # Prediksi dengan model Farhan
        result_farhan = model_farhan.predict(input_data)[0]
        prediction_farhan = label_mapping.get(result_farhan, "Unknown")


    except Exception as e:
        print("TIPE model_abid:", type(model_abid))
        print("TIPE model_farhan:", type(model_farhan))
        prediction_abid = f"Error: {str(e)}"
        prediction_farhan = f"Error: {str(e)}"

    return render_template("model_both.html",
                           prediction_abid=prediction_abid,
                           prediction_farhan=prediction_farhan)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', debug=True)
