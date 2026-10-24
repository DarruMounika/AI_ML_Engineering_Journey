
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 1. Create a sample monthly sales dataset
data = {
    "month": list(range(1, 25)),
    "advertising_spend": [
        1000, 1100, 1050, 1200, 1250, 1300,
        1280, 1400, 1450, 1500, 1550, 1600,
        1580, 1700, 1750, 1800, 1850, 1900,
        1880, 2000, 2050, 2100, 2200, 2300
    ],
    "average_price": [
        50, 50, 52, 51, 50, 49,
        52, 51, 50, 50, 49, 48,
        50, 49, 48, 50, 49, 48,
        49, 48, 47, 48, 47, 46
    ],
    "monthly_sales": [
        200, 215, 210, 235, 250, 260,
        255, 285, 300, 310, 330, 345,
        340, 370, 385, 400, 420, 435,
        430, 460, 480, 495, 520, 545
    ]
}

df = pd.DataFrame(data)

print("=== Original Dataset ===")
print(df.to_string(index=False))

# 2. Separate features (X) and target (y)
X = df[["month", "advertising_spend", "average_price"]]
y = df["monthly_sales"]

print("\nFeature columns:", list(X.columns))
print("Target column:", y.name)

# 3. Split chronologically: first 80% for training,
#    last 20% for testing
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 4. Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel training completed!")

# 5. Predict sales for the test period
predictions = model.predict(X_test)

results = pd.DataFrame({
    "Month": X_test["month"].values,
    "Actual_Sales": y_test.values,
    "Predicted_Sales": predictions.round(2)
})

print("\n=== Actual vs Predicted Sales ===")
print(results.to_string(index=False))

# 6. Evaluate the model using MAE
mae = mean_absolute_error(y_test, predictions)

print(f"\nMean Absolute Error (MAE): {mae:.2f} sales units")


# Simple baseline: predict each month using
# the previous month's actual sales
baseline_predictions = df["monthly_sales"].iloc[
    split_index - 1 : len(df) - 1
].values

baseline_mae = mean_absolute_error(y_test, baseline_predictions)

print(f"ML Model MAE: {mae:.2f}")
print(f"Baseline MAE:  {baseline_mae:.2f}")

if mae < baseline_mae:
    print("The ML model performs better than the baseline.")
else:
    print("The baseline performs as well as or better than the ML model.")