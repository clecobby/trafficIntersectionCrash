import numpy as np


# ---------------------------------
# 1. Load data
# ---------------------------------

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
# 2. Scale features
# ---------------------------------

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std


# ---------------------------------
# 3. Sigmoid
# ---------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------
# 4. Initialize model
# ---------------------------------

w = np.zeros(4)
b = 0.0

learning_rate = 0.1
iterations = 1000

N = len(y_train)


# ---------------------------------
# 5. Training loop
# ---------------------------------

for i in range(iterations):

    # Forward pass
    z = np.dot(X_train_scaled, w) + b
    probabilities = sigmoid(z)

    # Loss
    epsilon = 1e-15

    loss = -np.mean(
        y_train * np.log(probabilities + epsilon)
        +
        (1 - y_train) * np.log(1 - probabilities + epsilon)
    )

    # Gradients
    errors = probabilities - y_train

    dw = np.dot(X_train_scaled.T, errors) / N
    db = np.sum(errors) / N

    # Gradient descent
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # Print progress every 100 iterations
    if i % 100 == 0:
        print("Iteration:", i, "Loss:", loss)


# ---------------------------------
# 6. Final learned parameters
# ---------------------------------

print("\nTraining complete")
print("Learned weights:", w)
print("Learned bias:", b)


# ---------------------------------
# 7. Convert weights back to
# original feature units
# ---------------------------------

w_original = w / std

b_original = b - np.sum(
    w * mean / std
)

print("\nParameters in original feature units:")
print("Weights:", w_original)
print("Bias:", b_original)


# True parameters used to generate data
true_w = np.array([
    0.0015,
    0.05,
    1.0,
    0.7
])

true_b = -6

print("\nTrue hidden parameters:")
print("Weights:", true_w)
print("Bias:", true_b)




# ---------------------------------
# 8. Evaluate on unseen test data
# ---------------------------------

z_test = np.dot(X_test_scaled, w) + b

test_probabilities = sigmoid(z_test)


# Test log loss
test_loss = -np.mean(
    y_test * np.log(test_probabilities + epsilon)
    +
    (1 - y_test) * np.log(
        1 - test_probabilities + epsilon
    )
)


print("\nGeneralization evaluation:")
print("Training loss:", loss)
print("Test loss:", test_loss)

final_train_probabilities = sigmoid(
    np.dot(X_train_scaled, w) + b
)

final_train_loss = -np.mean(
    y_train * np.log(
        final_train_probabilities + epsilon
    )
    +
    (1 - y_train) * np.log(
        1 - final_train_probabilities + epsilon
    )
)

print("\nGeneralization evaluation:")
print("Final training loss:", final_train_loss)
print("Test loss:", test_loss)



# ---------------------------------
# 9. Convert probabilities to classes
# ---------------------------------

threshold = 0.2

y_pred = (
    test_probabilities >= threshold
).astype(int)


# ---------------------------------
# Confusion matrix values
# ---------------------------------

TP = np.sum((y_pred == 1) & (y_test == 1))
TN = np.sum((y_pred == 0) & (y_test == 0))
FP = np.sum((y_pred == 1) & (y_test == 0))
FN = np.sum((y_pred == 0) & (y_test == 1))


print("\nConfusion Matrix:")
print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)


accuracy = (TP + TN) / len(y_test)

print("\nAccuracy:", accuracy)
print("Accuracy %:", accuracy * 100)

recall = TP / (TP + FN)

print("Recall:", recall)
print("Recall %:", recall * 100)


precision = TP / (TP + FP)

print("Precision:", precision)
print("Precision %:", precision * 100)


f1 = 2 * (
    precision * recall
) / (
    precision + recall
)

print("F1 score:", f1)