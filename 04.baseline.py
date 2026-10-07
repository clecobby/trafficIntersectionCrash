import numpy as np

# Load the train/test split produced by 03.splitdata.py
# instead of regenerating and reshuffling the data here
train_data = np.loadtxt("data/train.csv", delimiter=",", skiprows=1)
test_data = np.loadtxt("data/test.csv", delimiter=",", skiprows=1)

X_train = train_data[:, :4]
y_train = train_data[:, 4].astype(int)

X_test = test_data[:, :4]
y_test = test_data[:, 4].astype(int)

# ---------------------------------
# Dumb baseline:
# predict NO CRASH for everyone
# ---------------------------------

y_pred = np.zeros(len(y_test), dtype=int)

# Compare prediction against truth
correct = np.sum(y_pred == y_test)

accuracy = correct / len(y_test)

print("Test examples:", len(y_test))
print("Correct predictions:", correct)
print("Wrong predictions:", len(y_test) - correct)
print("Accuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100)
