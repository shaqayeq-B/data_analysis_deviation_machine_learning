import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("part_deviation_data.csv")

X = df[['Expected']]
y = df['Actual']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

df['Predicted'] = y_pred
df['Deviation'] = y - y_pred

print(df.head())
print("Mean deviation:", df['Deviation'].mean())
print("Standard deviation of deviation:", df['Deviation'].std())
print("Mean Squared Error (MSE):", mean_squared_error(y, y_pred))

df.to_csv("output_with_deviation.csv", index=False)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', linewidth=2, label='Predicted')
plt.title("Actual vs Predicted with Linear Regression")
plt.xlabel("Expected")
plt.ylabel("Actual")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(df['Deviation'], marker='o', linestyle='-', color='green')
plt.title("Deviation (Actual - Predicted)")
plt.xlabel("Index")
plt.ylabel("Deviation")
plt.grid(True)
plt.show()