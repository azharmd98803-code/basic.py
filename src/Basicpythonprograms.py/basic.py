import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------
# Global variables
# -------------------
data = None
model = None


def pause():
    input("\nPress Enter to continue...")


# -------------------
# Load Dataset
# -------------------
def load_dataset():
    global data
    try:
        data = pd.read_csv("src/Basicpythonprograms.py/student_data.csv")
        print("\nDataset Loaded Successfully!")
    except FileNotFoundError:
        print("\nError: student_data.csv not found!")
    pause()


# -------------------
# Dataset Info
# -------------------
def dataset_info():
    if data is None:
        print("\nLoad dataset first!")
        pause()
        return

    print("\nDataset Info:")
    print(data.info())
    print("\nFirst 5 Rows:")
    print(data.head())
    pause()


# -------------------
# Clean Data
# -------------------
def clean_data():
    global data
    if data is None:
        print("\nLoad dataset first!")
        pause()
        return

    print("\nDATA BEFORE CLEANING:")
    print(data)

    data.dropna(inplace=True)

    print("\nDATA AFTER CLEANING:")
    print(data)
    print("\nData cleaned successfully!")
    pause()


# -------------------
# Statistical Summary
# -------------------
def statistical_summary():
    if data is None:
        print("\nLoad dataset first!")
        pause()
        return

    print("\nStatistical Summary:")
    print(data.describe())
    pause()


# -------------------
# Visualizations
# -------------------
def visualization_menu():
    if data is None:
        print("\nLoad dataset first!")
        pause()
        return

    while True:
        print("\nVISUALIZATION MENU")
        print("1. Study Hours vs Marks")
        print("2. Attendance vs Marks")
        print("3. Correlation Heatmap")
        print("4. Back")

        choice = input("Enter choice: ")

        if choice == "1":
            sns.scatterplot(x="study_hours", y="marks", data=data)
            plt.title("Study Hours vs Marks")
            plt.show()
            pause()

        elif choice == "2":
            sns.scatterplot(x="attendance", y="marks", data=data)
            plt.title("Attendance vs Marks")
            plt.show()
            pause()

        elif choice == "3":
            sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm")
            plt.title("Correlation Heatmap")
            plt.show()
            pause()

        elif choice == "4":
            break
        else:
            print("Invalid choice!")
            pause()


# -------------------
# Train ML Model
# -------------------
def train_model():
    global model

    if data is None:
        print("\nLoad dataset first!")
        pause()
        return

    X = data[["study_hours", "attendance", "previous_score"]]
    y = data["marks"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nModel Trained Successfully!")
    print("MAE:", mean_absolute_error(y_test, predictions))
    print("R2 Score:", r2_score(y_test, predictions))
    pause()


# -------------------
# Predict Student Score
# -------------------
def predict_score():
    if model is None:
        print("\nTrain model first!")
        pause()
        return

    try:
        study_hours = float(input("Enter Study Hours: "))
        attendance = float(input("Enter Attendance %: "))
        previous_score = float(input("Enter Previous Score: "))

        input_data = pd.DataFrame(
            [[study_hours, attendance, previous_score]],
            columns=["study_hours", "attendance", "previous_score"]
        )

        prediction = model.predict(input_data)
        print(f"\nPredicted Student Score: {prediction[0]:.2f}")

    except ValueError:
        print("\nInvalid input! Enter numeric values only.")

    pause()


# -------------------
# Main Menu
# -------------------
def main_menu():
    while True:
        print("\n===================================")
        print("STUDENT PERFORMANCE ANALYTICS SYSTEM")
        print("===================================")
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
            load_dataset()
        elif choice == "2":
            dataset_info()
        elif choice == "3":
            clean_data()
        elif choice == "4":
            statistical_summary()
        elif choice == "5":
            visualization_menu()
        elif choice == "6":
            train_model()
        elif choice == "7":
            predict_score()
        elif choice == "8":
            print("\nThank you 😊")
            sys.exit()
        else:
            print("\nInvalid choice!")
            pause()


# -------------------
# Run Program
# -------------------
main_menu()