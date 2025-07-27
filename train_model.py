# train_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Sample training data
data = {
    "size": [1000, 1500, 2000, 2500, 3000],
    "price": [50, 75, 100, 125, 150]
}
df = pd.DataFrame(data)

X = df[["size"]]
y = df["price"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model to model.pkl
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
