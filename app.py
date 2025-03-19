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
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('model_abid.pkl', 'rb') as model_file_abid:
    model_abid = pickle.load(model_file_abid)

# Label Mapping for Predictions
label_mapping = {0: "Non Diabetes", 1: "Pre-diabetes", 2: "Diabetes"}

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict1")
def predict():
    return render_template("model_farhan.html")

@app.route("/predictFarhan", methods=['GET', 'POST'])
def predictfarhan():
    prediction = None
    if request.method == 'POST':
        try:
            Gender = request.form.get('Gender')
            Age = request.form.get('Age', type=int)
            Urea = request.form.get('Urea', type=float)
            Creatine = request.form.get('Creatine', type=float)
            HbA1c = request.form.get('HbA1c', type=float)
            Cholesterol = request.form.get('Cholesterol', type=float)
            Triglycerides = request.form.get('Trigliserida', type=float)
            VLDL = request.form.get('VLDL', type=float)
            BMI = request.form.get('BMI', type=float)
            
            # Convert Gender to numerical value
            Gender_encoded = 1 if Gender.lower() == "male" else 0

            # Create input DataFrame
            input_data = pd.DataFrame([[Gender_encoded, Age, Urea, Creatine, HbA1c, Cholesterol, Triglycerides, VLDL, BMI]],
                                      columns=['Gender', 'Age', 'Urea', 'Creatine', 'HbA1c', 'Cholesterol', 'Trigliserida', 'VLDL', 'BMI'])

            # Predict
            prediction_raw = model.predict(input_data)[0]
            prediction = label_mapping.get(prediction_raw, "Unknown")

            # Save to database
            new_entry = TA(Gender=Gender, Age=Age, HbA1c=HbA1c, Cholesterol=Cholesterol,
                           Triglycerides=Triglycerides, BMI=BMI, Prediction=prediction)
            db.session.add(new_entry)
            db.session.commit()

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("model_farhan.html", prediction=prediction)

@app.route("/predict2")
def prediction():
    return render_template("model_abid.html")

@app.route("/predictAbid", methods=['GET', 'POST'])
def predict_abid():
    error = None
    prediction = None

    if request.method == 'POST':
        try:
            Gender = request.form.get('Gender')
            Age = request.form.get('Age', type=int)
            Urea = request.form.get('Urea', type=float)
            Creatine = request.form.get('Creatine', type=float)
            HbA1c = request.form.get('HbA1c', type=float)
            Cholesterol = request.form.get('Cholesterol', type=float)
            Triglycerides = request.form.get('Trigliserida', type=float)
            VLDL = request.form.get('VLDL', type=float)
            BMI = request.form.get('BMI', type=float)

            if None in [Gender, Age, Urea, Creatine, HbA1c, Cholesterol, Triglycerides, VLDL, BMI]:
                raise ValueError("All fields are required.")

            Gender_encoded = 1 if Gender.lower() == "male" else 0

            input_data = pd.DataFrame([[Gender_encoded, Age, Urea, Creatine, HbA1c, Cholesterol, Triglycerides, VLDL, BMI]],
                                      columns=['Gender', 'Age', 'Urea', 'Creatine', 'HbA1c', 'Cholesterol', 'Trigliserida', 'VLDL', 'BMI'])

            # Predict
            prediction_raw = model_abid.predict(input_data)[0]
            prediction = label_mapping.get(prediction_raw, "Unknown")

            # Save to database
            new_entry = ta(Gender=Gender, Age=Age, Urea=Urea, Creatine=Creatine, HbA1c=HbA1c, 
                           Cholesterol=Cholesterol, Triglycerides=Triglycerides, VLDL=VLDL, BMI=BMI, 
                           Prediction=prediction)
            db.session.add(new_entry)
            db.session.commit()

        except Exception as e:
            error = str(e)

    return render_template("model_abid.html", prediction=prediction, error=error)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', debug=True)
