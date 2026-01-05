# Student Performance Analysis Using Data Science

# 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2. Create sample student dataset
data = {
    'Attendance': [85, 70, 90, 60, 95, 80, 75, 88, 92, 65],
    'Study_Hours': [3, 2, 4, 1, 5, 3, 2, 4, 5, 2],
    'Internal_Marks': [18, 14, 20, 12, 22, 17, 15, 19, 21, 13],
    'Final_Result': [75, 60, 85, 50, 90, 72, 65, 80, 88, 55]
}

df = pd.DataFrame(data)

print("Student Dataset:\n")
print(df)

# 3. Dataset information
print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# 4. Data Visualization
plt.figure()
plt.scatter(df['Attendance'], df['Final_Result'])
plt.xlabel("Attendance (%)")
plt.ylabel("Final Result")
plt.title("Attendance vs Final Result")
plt.show()

plt.figure()
plt.scatter(df['Study_Hours'], df['Final_Result'])
plt.xlabel("Study Hours")
plt.ylabel("Final Result")
plt.title("Study Hours vs Final Result")
plt.show()

# 5. Prepare features and target
X = df[['Attendance', 'Study_Hours', 'Internal_Marks']]
y = df['Final_Result']

# 6. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Apply Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 8. Predict results
y_pred = model.predict(X_test)

print("\nPredicted Results:", y_pred)
print("Actual Results:", y_test.values)

# 9. Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R2 Score:", r2)

print("\nStudent Performance Analysis Completed Successfully")