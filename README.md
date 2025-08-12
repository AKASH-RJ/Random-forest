# Student Performance Prediction using Random Forest & Flask

## Overview

This project predicts **student exam performance** based on various academic and personal attributes. The machine learning model is built using **Random Forest Regressor** (from scikit-learn) and is deployed as a **Flask web application** with HTML & CSS for the interface.

-----

## Features

  - **Random Forest Regression** for better accuracy compared to a single decision tree.
  - **Flask backend** for handling prediction requests.
  - **Interactive HTML form** to input student details.
  - **CSS styling** for a clean and user-friendly UI.
  - **CSV dataset** with 50 student records for training.

-----

## Project Structure

```
student_performance_rf/
│
├── model.py             # Trains and saves the Random Forest model
├── app.py               # Flask application for serving predictions
├── templates/
│   ├── index.html       # Main input form page
│   └── result.html      # Displays prediction results
├── static/
│   └── style.css        # CSS for frontend styling
├── dataset.csv          # Sample dataset (50 students)
├── requirements.txt     # Required Python dependencies
└── README.md            # Project documentation
```

-----

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:

```
Flask==3.0.0
pandas==2.1.4
scikit-learn==1.3.2
numpy==1.26.2
```

-----

## Dataset

The dataset (`dataset.csv`) contains student details and their final exam score.
Example:

```
hours_studied,attendance_percentage,assignments_completed,exam_score
5,85,10,75
3,70,8,60
8,95,12,88
```

Features:

  - `hours_studied`: Number of hours a student studies per week.
  - `attendance_percentage`: Class attendance rate.
  - `assignments_completed`: Number of assignments completed.
  - `exam_score`: Final score in the exam (Target variable).

-----

## How It Works

### Model Training (`model.py`)

  - Loads `dataset.csv` into a Pandas DataFrame.
  - Splits data into training and testing sets.
  - Trains a Random Forest Regressor.
  - Saves the model as `model.pkl`.

### Web Application (`app.py`)

  - Loads the saved `model.pkl`.
  - Accepts input from an HTML form.
  - Predicts student exam score.
  - Displays the result on a new page.

-----

## Running the Project

1.  **Train the Model**
    ```bash
    python model.py
    ```
2.  **Run Flask App**
    ```bash
    python app.py
    ```
3.  **Open Browser**
    Go to: `http://127.0.0.1:5000/`

-----

## Screenshots
---
Home Page

<img width="507" height="397" alt="Screenshot 2025-08-12 121618" src="https://github.com/user-attachments/assets/77ffc11b-ba9a-49f5-86e6-a2456f24d858" />

---
Prediction Result

<img width="557" height="482" alt="Screenshot 2025-08-12 121628" src="https://github.com/user-attachments/assets/f04ca84f-1eec-45d4-9f1c-97a82a53a35e" />
