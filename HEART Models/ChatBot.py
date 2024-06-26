import tkinter as tk
from tkinter import messagebox, ttk
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the saved model, scaler, and imputer
with open('HeartDiseaseModel.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('imputer.pkl', 'rb') as file:
    imputer = pickle.load(file)

with open('X_test.pkl', 'rb') as file:
    X_test = pickle.load(file)

# Define the selected features for prediction (modify based on your best model)
selected_features = ['age', 'sysBP', 'glucose', 'totChol', 'cigsPerDay', 'diaBP', 'prevalentHyp', 'diabetes', 'BPMeds', 'male']

# Function to get user input and make a prediction
def predict_heart_disease():
    try:
        # Collect user inputs
        inputs = [
            age_var.get(),
            bp_var.get(),
            glucose_var.get(),
            chol_var.get(),
            cigs_var.get(),
            dia_bp_var.get(),
            1 if hyp_var.get() == 'Yes' else 0,
            1 if diabetes_var.get() == 'Yes' else 0,
            1 if bp_med_var.get() == 'Yes' else 0,
            1 if gender_var.get() == 'Male' else 0
        ]

        # Convert inputs to a DataFrame and ensure correct feature order
        inputs_df = pd.DataFrame([inputs], columns=selected_features)
        
        # Handle missing values using the imputer and transform inputs using the scaler
        inputs_imputed = imputer.transform(inputs_df)
        inputs_scaled = scaler.transform(inputs_imputed)

        # Make prediction
        prediction = model.predict(inputs_scaled)[0]
        probability = model.predict_proba(inputs_scaled)[0][1] * 100

        if prediction == 1:
            result = "Yes"
            message = f"Heart Disease: Yes\n\nProbability: {probability:.2f}%\n\nIt is recommended to consult a healthcare professional for further evaluation and potential treatment. Additionally, consider the following lifestyle changes:\n\n- Maintain a healthy diet\n- Exercise regularly\n- Quit smoking\n- Limit alcohol consumption\n- Monitor blood pressure and blood sugar levels\n- Manage stress effectively"
        else:
            result = "No"
            message = f"Heart Disease: No\n\nProbability: {probability:.2f}%\n\nContinue maintaining a healthy lifestyle to keep your heart in good condition. Regular check-ups are still recommended."

        messagebox.showinfo("Prediction", message)
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to test random cases from X_test
def test_random_case():
    try:
        # Select a random case from X_test
        random_case = X_test.sample(1)
        random_case_values = random_case.values
        
        # Impute and scale the random case
        random_case_imputed = imputer.transform(random_case_values)
        random_case_scaled = scaler.transform(random_case_imputed)

        # Make prediction
        prediction = model.predict(random_case_scaled)[0]
        probability = model.predict_proba(random_case_scaled)[0][1] * 100

        # Display the selected case values (questions with answers)
        message = "Random Test Case:\n\n"
        message += f"Age: {random_case['age'].values[0]}\n"
        message += f"Systolic Blood Pressure: {random_case['sysBP'].values[0]}\n"
        message += f"Diastolic Blood Pressure: {random_case['diaBP'].values[0]}\n"
        message += f"Glucose level: {random_case['glucose'].values[0]}\n"
        message += f"Total Cholesterol level: {random_case['totChol'].values[0]}\n"
        message += f"Cigarettes per day: {random_case['cigsPerDay'].values[0]}\n"
        message += f"Hypertension: {'Yes' if random_case['prevalentHyp'].values[0] == 1 else 'No'}\n"
        message += f"Diabetes: {'Yes' if random_case['diabetes'].values[0] == 1 else 'No'}\n"
        message += f"Blood Pressure Medication: {'Yes' if random_case['BPMeds'].values[0] == 1 else 'No'}\n"
        message += f"Gender: {'Male' if random_case['male'].values[0] == 1 else 'Female'}\n"

        if prediction == 1:
            result = "Yes"
            message += f"\nHeart Disease: Yes\n\nProbability: {probability:.2f}%"
        else:
            result = "No"
            message += f"\nHeart Disease: No\n\nProbability: {probability:.2f}%"

        messagebox.showinfo("Random Test Case", message)
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Initialize the GUI application
app = tk.Tk()
app.title("Heart Disease Prediction / توقع أمراض القلب")

# Create input fields based on selected features
tk.Label(app, text="Age / العمر: How old are you? / كم عمرك؟").grid(row=0, column=0)
age_var = tk.IntVar()
tk.Entry(app, textvariable=age_var).grid(row=0, column=1)

tk.Label(app, text="Systolic Blood Pressure / ضغط الدم الانقباضي:").grid(row=1, column=0)
bp_var = tk.DoubleVar()
tk.Entry(app, textvariable=bp_var).grid(row=1, column=1)

tk.Label(app, text="Diastolic Blood Pressure / ضغط الدم الانبساطي:").grid(row=2, column=0)
dia_bp_var = tk.DoubleVar()
tk.Entry(app, textvariable=dia_bp_var).grid(row=2, column=1)

tk.Label(app, text="Glucose level / مستوى الجلوكوز:").grid(row=3, column=0)
glucose_var = tk.DoubleVar()
tk.Entry(app, textvariable=glucose_var).grid(row=3, column=1)

tk.Label(app, text="Total Cholesterol level / مستوى الكوليسترول الكلي:").grid(row=4, column=0)
chol_var = tk.DoubleVar()
tk.Entry(app, textvariable=chol_var).grid(row=4, column=1)

tk.Label(app, text="Cigarettes per day / عدد السجائر في اليوم:").grid(row=5, column=0)
cigs_var = tk.DoubleVar()
tk.Entry(app, textvariable=cigs_var).grid(row=5, column=1)

tk.Label(app, text="Hypertension / الارتفاع في ضغط الدم: Are you hypertensive? (Yes/No) / هل تعاني من ارتفاع ضغط الدم؟ (نعم/لا)").grid(row=6, column=0)
hyp_var = tk.StringVar()
hyp_combo = ttk.Combobox(app, textvariable=hyp_var)
hyp_combo['values'] = ('Yes', 'No')
hyp_combo.grid(row=6, column=1)

tk.Label(app, text="Diabetes / السكري: Do you have diabetes? (Yes/No) / هل تعاني من مرض السكري؟ (نعم/لا)").grid(row=7, column=0)
diabetes_var = tk.StringVar()
diabetes_combo = ttk.Combobox(app, textvariable=diabetes_var)
diabetes_combo['values'] = ('Yes', 'No')
diabetes_combo.grid(row=7, column=1)

tk.Label(app, text="Blood Pressure Medication / الأدوية لضغط الدم: Are you on blood pressure medication? (Yes/No) / هل تتناول أدوية لضغط الدم؟ (نعم/لا)").grid(row=8, column=0)
bp_med_var = tk.StringVar()
bp_med_combo = ttk.Combobox(app, textvariable=bp_med_var)
bp_med_combo['values'] = ('Yes', 'No')
bp_med_combo.grid(row=8, column=1)

tk.Label(app, text="Gender / الجنس: What is your gender? (Male/Female) / ما هو جنسك؟ (ذكر/أنثى)").grid(row=9, column=0)
gender_var = tk.StringVar()
gender_combo = ttk.Combobox(app, textvariable=gender_var)
gender_combo['values'] = ('Male', 'Female')
gender_combo.grid(row=9, column=1)

# Create Predict button
tk.Button(app, text="Predict / توقع", command=predict_heart_disease).grid(row=10, column=0, columnspan=2)

# Create Test Random Case button
tk.Button(app, text="Test Random Case / اختبار حالة عشوائية", command=test_random_case).grid(row=11, column=0, columnspan=2)

# Start the GUI application
app.mainloop()
