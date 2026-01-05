# ============================================================
# SMART DATA SCIENCE & ANALYTICS TOOLKIT
# Author: Ajju
# Level: Advanced / Faculty-Ready / Industry Style
# ============================================================

import os
import sys
import time
import json
import math
import warnings
import logging

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# LOGGING CONFIG
# ============================================================

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ============================================================
# GLOBAL VARIABLES
# ============================================================

dataset = None
target_column = None
model = None
scaler = None

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def slow_print(text, delay=0.01):
    for char in text:
        print(char, end="")
        time.sleep(delay)
    print()

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def pause():
    input("\nPress Enter to continue...")

# ============================================================
# DATA LOADING MODULE
# ============================================================

def load_dataset():
    global dataset
    clear_screen()
    slow_print("📂 DATASET LOADER")

    path = input("Enter CSV file path: ")

    try:
        dataset = pd.read_csv(path)
        logging.info("Dataset loaded successfully")
        slow_print("✅ Dataset loaded successfully!")
        slow_print(f"Rows: {dataset.shape[0]} | Columns: {dataset.shape[1]}")
    except Exception as e:
        logging.error(str(e))
        slow_print("❌ Error loading dataset")

    pause()

# ============================================================
# DATA PREVIEW MODULE
# ============================================================

def preview_data():
    clear_screen()
    if dataset is None:
        slow_print("❌ No dataset loaded")
        pause()
        return

    slow_print("🔍 DATA PREVIEW")
    print(dataset.head(10))
    pause()

# ============================================================
# DATA CLEANING MODULE
# ============================================================

def clean_data():
    global dataset
    clear_screen()

    if dataset is None:
        slow_print("❌ Load dataset first")
        pause()
        return

    slow_print("🧹 DATA CLEANING")

    slow_print("1. Remove missing values")
    slow_print("2. Fill missing values (mean)")
    slow_print("3. Remove duplicates")

    choice = input("Select option: ")

    if choice == "1":
        dataset.dropna(inplace=True)
        slow_print("✅ Missing values removed")

    elif choice == "2":
        dataset.fillna(dataset.mean(numeric_only=True), inplace=True)
        slow_print("✅ Missing values filled")

    elif choice == "3":
        dataset.drop_duplicates(inplace=True)
        slow_print("✅ Duplicates removed")

    else:
        slow_print("❌ Invalid choice")

    pause()

# ============================================================
# STATISTICAL ANALYSIS MODULE
# ============================================================

def statistical_summary():
    clear_screen()

    if dataset is None:
        slow_print("❌ Load dataset first")
        pause()
        return

    slow_print("📊 STATISTICAL SUMMARY")
    print(dataset.describe())
    pause()

# ============================================================
# VISUALIZATION MODULE
# ============================================================

def visualization_menu():
    clear_screen()

    if dataset is None:
        slow_print("❌ Load dataset first")
        pause()
        return

    slow_print("📈 DATA VISUALIZATION")
    slow_print("1. Histogram")
    slow_print("2. Boxplot")
    slow_print("3. Correlation Heatmap")

    choice = input("Select: ")

    if choice == "1":
        column = input("Enter column name: ")
        dataset[column].hist()
        plt.show()

    elif choice == "2":
        column = input("Enter column name: ")
        sns.boxplot(x=dataset[column])
        plt.show()

    elif choice == "3":
        sns.heatmap(dataset.corr(), annot=True, cmap="coolwarm")
        plt.show()

    else:
        slow_print("❌ Invalid choice")

    pause()

# ============================================================
# FEATURE ENGINEERING MODULE
# ============================================================

def feature_engineering():
    global dataset
    clear_screen()

    if dataset is None:
        slow_print("❌ Load dataset first")
        pause()
        return

    slow_print("⚙️ FEATURE ENGINEERING")

    for col in dataset.select_dtypes(include="object").columns:
        encoder = LabelEncoder()
        dataset[col] = encoder.fit_transform(dataset[col])

    slow_print("✅ Categorical encoding completed")
    pause()

# ============================================================
# MODEL TRAINING MODULE
# ============================================================

def train_model():
    global model, target_column, scaler
    clear_screen()

    if dataset is None:
        slow_print("❌ Load dataset first")
        pause()
        return

    slow_print("🤖 MODEL TRAINING")
    slow_print("1. Linear Regression")
    slow_print("2. Logistic Regression")

    choice = input("Select model: ")
    target_column = input("Enter target column: ")

    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if choice == "1":
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        slow_print(f"MSE: {mean_squared_error(y_test, predictions)}")
        slow_print(f"R2 Score: {r2_score(y_test, predictions)}")

    elif choice == "2":
        model = LogisticRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        slow_print(f"Accuracy: {accuracy_score(y_test, predictions)}")
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))

    else:
        slow_print("❌ Invalid model")

    pause()

# ============================================================
# PREDICTION MODULE
# ============================================================

def predict():
    clear_screen()

    if model is None:
        slow_print("❌ Train a model first")
        pause()
        return

    slow_print("🔮 PREDICTION SYSTEM")

    values = []
    for col in dataset.drop(columns=[target_column]).columns:
        val = float(input(f"Enter value for {col}: "))
        values.append(val)

    values = scaler.transform([values])
    result = model.predict(values)

    slow_print(f"✅ Prediction Result: {result[0]}")
    pause()

# ============================================================
# REPORT GENERATOR
# ============================================================

def save_report():
    clear_screen()

    if dataset is None:
        slow_print("❌ No dataset")
        pause()
        return

    report = {
        "rows": int(dataset.shape[0]),
        "columns": int(dataset.shape[1]),
        "columns_list": list(dataset.columns)
    }

    with open("report.json", "w") as f:
        json.dump(report, f, indent=4)

    slow_print("📄 Report saved as report.json")
    pause()

# ============================================================
# MAIN MENU
# ============================================================

def main_menu():
    while True:
        clear_screen()
        slow_print("🔥 SMART DATA SCIENCE TOOLKIT 🔥\n")

        slow_print("1. Load Dataset")
        slow_print("2. Preview Data")
        slow_print("3. Clean Data")
        slow_print("4. Statistical Summary")
        slow_print("5. Visualization")
        slow_print("6. Feature Engineering")
        slow_print("7. Train Model")
        slow_print("8. Predict")
        slow_print("9. Save Report")
        slow_print("0. Exit")

        choice = input("\nEnter choice: ")

        if choice == "1":
            load_dataset()
        elif choice == "2":
            preview_data()
        elif choice == "3":
            clean_data()
        elif choice == "4":
            statistical_summary()
        elif choice == "5":
            visualization_menu()
        elif choice == "6":
            feature_engineering()
        elif choice == "7":
            train_model()
        elif choice == "8":
            predict()
        elif choice == "9":
            save_report()
        elif choice == "0":
            slow_print("👋 Exiting...")
            sys.exit()
        else:
            slow_print("❌ Invalid choice")
            pause()

# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main_menu()