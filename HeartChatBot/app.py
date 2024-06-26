import json
import os

import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError

app = Flask(__name__)

# Load all the saved models
loaded_models = {}
model_files = [
    'Models/HeartDiseaseModelLogisticRegression.pkl',
    # 'Models/HeartDiseaseModelDecisionTree.pkl',
    # 'Models/HeartDiseaseModelGaussianNB.pkl',
    # 'Models/HeartDiseaseModelKNN.pkl',
    # 'Models/HeartDiseaseModelRandomForest.pkl',
    # 'Models/HeartDiseaseModelSVM.pkl'
]

for model_file in model_files:
    with open(model_file, 'rb') as file:
        loaded_models[model_file.split('/')[-1].split('.')[0]] = pickle.load(file)

# Load the scaler
with open('Models/MinMaxScaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the questions for the form
questions = [
    {"label": "Systolic Blood Pressure", "name": "sysBP"},
    {"label": "Glucose", "name": "glucose"},
    {"label": "Age", "name": "age"},
    {"label": "Total Cholesterol", "name": "totChol"},
    {"label": "Cigarettes per Day", "name": "cigsPerDay"},
    {"label": "Diastolic Blood Pressure", "name": "diaBP"},
    {"label": "Prevalent Hypertension", "name": "prevalentHyp"},
    {"label": "Diabetes", "name": "diabetes"},
    {"label": "Blood Pressure Medication", "name": "BPMeds"},
    {"label": "Gender", "name": "male"}
]

# Homepage route
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return redirect(url_for('predict'))
    return render_template('index.html', questions=questions)

# Endpoint to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()

            # Extract answers from the POST request
            user_inputs = {}
            for answer in data['answers']:
                user_inputs[answer['question']] = answer['answer']
            uestions = [
                {"label": "Age / العمر", "description": "How old are you? / كم عمرك؟", "name": "age", "type": "text"},
                {"label": "Systolic Blood Pressure / ضغط الدم الانقباضي",
                 "description": "What is your systolic blood pressure? / ما هو ضغط الدم الانقباضي لديك؟",
                 "name": "sysBP", "type": "text"},
                {"label": "Diastolic Blood Pressure / ضغط الدم الانبساطي",
                 "description": "What is your diastolic blood pressure? / ما هو ضغط الدم الانبساطي لديك؟",
                 "name": "diaBP", "type": "text"},
                {"label": "Glucose level / مستوى الجلوكوز",
                 "description": "What is your glucose level? / ما هو مستوى الجلوكوز لديك؟", "name": "glucose",
                 "type": "text"},
                {"label": "Total Cholesterol level / مستوى الكوليسترول الكلي",
                 "description": "What is your total cholesterol level? / ما هو مستوى الكوليسترول الكلي لديك؟",
                 "name": "totChol", "type": "text"},
                {"label": "Cigarettes per day / عدد السجائر في اليوم",
                 "description": "How many cigarettes do you smoke per day? / كم عدد السجائر التي تدخنها في اليوم؟",
                 "name": "cigsPerDay", "type": "text"},
                {"label": "Hypertension / الارتفاع في ضغط الدم",
                 "description": "Are you hypertensive? (Yes/No) / هل تعاني من ارتفاع ضغط الدم؟ (نعم/لا)",
                 "name": "prevalentHyp", "type": "radio"},
                {"label": "Diabetes / السكري",
                 "description": "Do you have diabetes? (Yes/No) / هل تعاني من مرض السكري؟ (نعم/لا)", "name": "diabetes",
                 "type": "radio"},
                {"label": "Blood Pressure Medication / الأدوية لضغط الدم",
                 "description": "Are you on blood pressure medication? (Yes/No) / هل تتناول أدوية لضغط الدم؟ (نعم/لا)",
                 "name": "BPMeds", "type": "radio"},
                {"label": "Gender / الجنس",
                 "description": "What is your gender? (Male/Female) / ما هو جنسك؟ (ذكر/أنثى)", "name": "gender",
                 "type": "radio"}
            ];
            # Prepare data for prediction
            # Prepare data for prediction
            new_data = np.array([
                [float(user_inputs['sysBP']),
                 float(user_inputs['glucose']),
                 float(user_inputs['age']),
                 float(user_inputs['totChol']),
                 float(user_inputs['cigsPerDay']),
                 float(user_inputs['diaBP']),
                 int(user_inputs['prevalentHyp']),
                 int(user_inputs['diabetes']),
                 int(user_inputs['BPMeds']),
                 int(user_inputs['gender'])]
            ])

            new_data_with_tenyearchd = np.concatenate((new_data, [[0]]),
                                                      axis=1)  # Assuming 'TenYearCHD' is 0 for this example

            # Scale the data
            new_data_scaled = scaler.transform(new_data_with_tenyearchd)
            new_data_scaled_10_features = new_data_scaled[:, :-1]

            # Make predictions with all models
            predictions = {}
            for model_name, model in loaded_models.items():
                prediction = model.predict(new_data_scaled_10_features)[0]
                predictions[model_name] = prediction

            return jsonify(predictions)

        except NotFittedError as nfe:
            error_message = "Scaler is not fitted. Please provide training data first."
            return jsonify({'error': error_message}), 500
        except ValueError as ve:
            error_message = f"ValueError: {str(ve)}"
            return jsonify({'error': error_message}), 400
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            return jsonify({'error': error_message}), 500

    else:
        return jsonify({'error': 'Method not allowed'}), 405


# Path to the JSON file
USER_DB = 'users.json'


def read_users():
    if not os.path.exists(USER_DB):
        with open(USER_DB, 'w') as f:
            json.dump([], f)
    with open(USER_DB, 'r') as f:
        return json.load(f)


def write_users(users):
    with open(USER_DB, 'w') as f:
        json.dump(users, f, indent=4)


@app.route('/')
@app.route('/login', methods=['GET'])
def show_login_form():
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def process_login():
    username = request.form.get('username')
    password = request.form.get('password')
    users = read_users()

    for user in users:
        if user['username'] == username and user['password'] == password:
            return redirect(url_for('index'))

    error = 'Invalid credentials. Please try again.'
    return render_template('login.html', error=error)


@app.route('/registration', methods=['GET'])
def show_registration_form():
    return render_template('registration.html')


@app.route('/registration', methods=['POST'])
def process_registration():
    fullname = request.form.get('fullname')
    username = request.form.get('username')
    password = request.form.get('password')
    users = read_users()

    for user in users:
        if user['username'] == username:
            error = 'Username already exists. Please try a different one.'
            return render_template('registration.html', error=error)

    new_user = {
        'fullname': fullname,
        'username': username,
        'password': password
    }
    users.append(new_user)
    write_users(users)

    return redirect(url_for('show_login_form'))


@app.route('/book', methods=['GET'])
def book():
    return render_template('books.html')

if __name__ == '__main__':
    app.run(debug=True)