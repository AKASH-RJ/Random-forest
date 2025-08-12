import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("ds.csv")

# Features & target
X = df[['study_hours', 'attendance_percentage', 'previous_score']]
y = df['final_grade'].map({'Pass': 1, 'Fail': 0})

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model & scaler
joblib.dump(model, "student_rf_model.pkl")
joblib.dump(scaler, "student_scaler.pkl")

print("Model and scaler saved successfully!")
