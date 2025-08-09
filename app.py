from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load dataset
df = pd.read_csv("random_forest.csv")

# Encode target
le = LabelEncoder()
df['Result'] = le.fit_transform(df['Result'])  # Pass=1, Fail=0

X = df[['Hours_Studied', 'Attendance_Percentage', 'Previous_Score']]
y = df['Result']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        hours = float(request.form['hours'])
        attendance = float(request.form['attendance'])
        prev_score = float(request.form['prev_score'])

        prediction = model.predict([[hours, attendance, prev_score]])[0]
        result = le.inverse_transform([prediction])[0]

        return render_template('index.html', prediction_text=f"Predicted Result: {result}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
