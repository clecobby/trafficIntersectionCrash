import numpy as np

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
# Feature scaling
# Learn mean/std from TRAINING data
# ---------------------------------

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

print("Feature means:")
print(mean)

print("Feature standard deviations:")
print(std)

print("First original training example:")
print(X_train[0])

print("First scaled training example:")
print(X_train_scaled[0])



w = np.zeros(4)
b = 0.0

print("Initial weights:", w)
print("Initial bias:", b)

# ---------------------------------
# Sigmoid function
# ---------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------
# Make predictions for ALL
# training examples
# ---------------------------------

z = np.dot(X_train_scaled, w) + b

probabilities = sigmoid(z)

print("First 10 probabilities:")
print(probabilities[:10])

print("First 10 actual outcomes:")
print(y_train[:10])


# ---------------------------------
# Binary cross-entropy loss
# ---------------------------------

epsilon = 1e-15

loss = -np.mean(
    y_train * np.log(probabilities + epsilon)
    +
    (1 - y_train) * np.log(1 - probabilities + epsilon)
)

print("Initial loss:", loss)



# ---------------------------------
# Calculate gradients
# ---------------------------------

N = len(y_train)

errors = probabilities - y_train

dw = np.dot(X_train_scaled.T, errors) / N
db = np.sum(errors) / N

print("Gradient for weights:")
print(dw)

print("Gradient for bias:")
print(db)



# ---------------------------------
# ONE gradient descent step
# ---------------------------------

learning_rate = 0.1

w = w - learning_rate * dw
b = b - learning_rate * db

print("\nAfter one gradient descent step:")
print("Updated weights:", w)
print("Updated bias:", b)


# ---------------------------------
# Recalculate predictions
# using UPDATED parameters
# ---------------------------------

z_new = np.dot(X_train_scaled, w) + b

probabilities_new = sigmoid(z_new)


loss_new = -np.mean(
    y_train * np.log(probabilities_new + epsilon)
    +
    (1 - y_train) * np.log(1 - probabilities_new + epsilon)
)

print("Old loss:", loss)
print("New loss:", loss_new)