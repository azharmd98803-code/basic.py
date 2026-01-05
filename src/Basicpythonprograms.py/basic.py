18# ============================================
# Student Performance Prediction & Analytics
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

# ============================================
# Global Variables
# ============================================

data = None
model = None

# ============================================
# Utility Functions
# ============================================

def line():
    print("=" * 60)

def pause():
    input("\nPress Enter to continue...")

# ============================================
# Load Dataset
# ============================================

def load_data():
    global data
    try:
        data = pd.read_csv("src/Basicpythonprograms.py/student_data.csv")
        print("\nDataset Loaded Successfully!")
    except FileNotFoundError:
        print("\nError: student_data.csv not found!")
        data = None

# ============================================
# Data Information
# ============================================

def data_info():
    if data is None:
        print("Load data first!")
        return
    line()
    print("DATASET INFORMATION")
    line()
    print(data.info())
    pause()

# ============================================
# Data Cleaning
# ============================================

def clean_data():
    global data
    if data is None:
        print("Load data first!")
        return
    line()
    print("DATA CLEANING")
    line()
    print("\nMissing values before cleaning:")
    print(data.isnull().sum())

    data.fillna(data.mean(numeric_only=True), inplace=True)

    print("\nMissing values after cleaning:")
    print(data.isnull().sum())
    pause()

# ============================================
# Statistical Summary
# ============================================

def statistical_summary():
    if data is None:
        print("Load data first!")
        return
    line()
    print("STATISTICAL SUMMARY")
    line()
    print(data.describe())
    pause()

# ============================================
# Visualizations
# ============================================

def plot_study_vs_score():
    sns.scatterplot(x="studyHours", y="finalScore", data=data)
    plt.title("Study Hours vs Final Score")
    plt.show()

def plot_attendance_vs_score():
    sns.scatterplot(x="attendance", y="finalScore", data=data)
    plt.title("Attendance vs Final Score")
    plt.show()

def plot_correlation_heatmap():
    plt.figure(figsize=(8,6))
    sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

def visual_menu():
    if data is None:
        print("Load data first!")
        return
    while True:
        line()
        print("VISUALIZATION MENU")
        line()
        print("1. Study Hours vs Score")
        print("2. Attendance vs Score")
        print("3. Correlation Heatmap")
        print("4. Back")
        ch = input("Enter choice: ")

        if ch == "1":
            plot_study_vs_score()
        elif ch == "2":
            plot_attendance_vs_score()
        elif ch == "3":
            plot_correlation_heatmap()
        elif ch == "4":
            break
        else:
            print("Invalid choice!")

# ============================================
# Model Training
# ============================================

def train_model():
    global model
    if data is None:
        print("Load data first!")
        return

    X = data[["StudyHours", "Attendance", "PreviousScore"]]
    y = data["FinalScore"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    line()
    print("MODEL PERFORMANCE")
    line()
    print("MAE :", mean_absolute_error(y_test, predictions))
    print("MSE :", mean_squared_error(y_test, predictions))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
    print("R2 Score:", r2_score(y_test, predictions))
    pause()

# ============================================
# Prediction System
# ============================================

def predict_score():
    if model is None:
        print("Train model first!")
        return

    try:
        study = float(input("Enter Study Hours: "))
        attendance = float(input("Enter Attendance %: "))
        prev = float(input("Enter Previous Score: "))

        result = model.predict([[study, attendance, prev]])
        print(f"\nPredicted Final Score: {result[0]:.2f}")
    except:
        print("Invalid input!")
    pause()

# ============================================
# Main Menu
# ============================================

def main_menu():
    while True:
        line()
        print("STUDENT PERFORMANCE ANALYTICS SYSTEM")
        line()
        print("1. Load Dataset")
        print("2. Dataset Info")
        print("3. Clean Data")
        print("4. Statistical Summary")
        print("5. Visualizations")
        print("6. Train ML Model")
        print("7. Predict Student Score")
        print("8. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            load_data()
        elif choice == "2":
            data_info()
        elif choice == "3":
            clean_data()
        elif choice == "4":
            statistical_summary()
        elif choice == "5":
            visual_menu()
        elif choice == "6":
            train_model()
        elif choice == "7":
            predict_score()
        elif choice == "8":
            print("Thank you!")
            sys.exit()
        else:
            print("Invalid choice!")

# ============================================
# Program Start
# ============================================

main_menu()


