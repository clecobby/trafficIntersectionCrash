import numpy as np

np.random.seed(42)
np.set_printoptions(suppress=True, precision=2)

N = 1000

# Generate features
traffic_volume = np.random.randint(200, 2501, N)
average_speed = np.random.uniform(15, 60, N)
rain = np.random.randint(0, 2, N)
night = np.random.randint(0, 2, N)

# Secret data-generating process
z = (
    -6
    + 0.0015 * traffic_volume
    + 0.05 * average_speed
    + 1.0 * rain
    + 0.7 * night
)

probability = 1 / (1 + np.exp(-z))
crash = np.random.binomial(1, probability)

# Build X and y
X = np.column_stack(
    (traffic_volume, average_speed, rain, night)
)

y = crash

# Shuffle indices
indices = np.arange(N)
np.random.shuffle(indices)

# 80% train, 20% test
split_index = int(0.8 * N)

train_indices = indices[:split_index]
test_indices = indices[split_index:]

X_train = X[train_indices]
y_train = y[train_indices]

X_test = X[test_indices]
y_test = y[test_indices]

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("\nTraining crash percentage:")
print(np.mean(y_train) * 100)

print("\nTesting crash percentage:")
print(np.mean(y_test) * 100)