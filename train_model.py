import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

# Dummy training data
X = np.array([
    [50, 2, 5],
    [80, 3, 10],
    [120, 4, 15]
])

y = np.array([150, 180, 220])

model = LinearRegression()
model.fit(X, y)

with open("ipl_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model created successfully!")