import numpy as np

# Load the SAME train/test data
train_data = np.loadtxt(
    "data/train.csv",
    delimiter=",",
    skiprows=1
)

test_data = np.loadtxt(
    "data/test.csv",
    delimiter=",",
    skiprows=1
)

X_train = train_data[:, :4]
y_train = train_data[:, 4].astype(int)

X_test = test_data[:, :4]
y_test = test_data[:, 4].astype(int)


# ---------------------------------
# Pick ONE observation
# ---------------------------------

x = X_test[0]

print("First test observation:", x)
print("Actual outcome:", y_test[0])


# ---------------------------------
# Manually supply weights
# ---------------------------------

w = np.array([
    0.0015,   # traffic volume
    0.05,     # average speed
    1.0,      # rain
    0.7       # night
])

b = -6


# ---------------------------------
# Linear score
# z = w^T x + b
# ---------------------------------

z = np.dot(w, x) + b

print("Linear score z:", z)


# ---------------------------------
# Sigmoid
# ---------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


probability = sigmoid(z)

print("Crash probability:", probability)
print("Crash probability %:", probability * 100)


# ---------------------------------
# OPTIONAL classification
# ---------------------------------

if probability >= 0.5:
    prediction = 1
else:
    prediction = 0

print("Predicted class:", prediction)
print("Actual class:", y_test[0])