from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("student_rf_model.pkl")
scaler = joblib.load("student_scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            study_hours = float(request.form['study_hours'])
            attendance_percentage = float(request.form['attendance_percentage'])
            previous_score = float(request.form['previous_score'])

            features = np.array([[study_hours, attendance_percentage, previous_score]])
            features_scaled = scaler.transform(features)
            pred = model.predict(features_scaled)[0]
            prediction = "Pass" if pred == 1 else "Fail"
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
