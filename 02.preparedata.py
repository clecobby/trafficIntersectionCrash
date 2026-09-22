import numpy as np

np.random.seed(42)

N = 1000

# Generate observations
traffic_volume = np.random.randint(200, 2501, N)
average_speed = np.random.uniform(15, 60, N)
rain = np.random.randint(0, 2, N)
night = np.random.randint(0, 2, N)

# ----- SECRET DATA-GENERATING PROCESS -----

z = (
    -6
    + 0.0015 * traffic_volume
    + 0.05 * average_speed
    + 1.0 * rain
    + 0.7 * night
)

probability = 1 / (1 + np.exp(-z))

crash = np.random.binomial(1, probability)

# ------------------------------------------
# From here onward, pretend we don't know
# anything above except the observations.
# ------------------------------------------

X = np.column_stack(
    (traffic_volume, average_speed, rain, night)
)

y = crash

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

print("\nFirst 5 X:")
print(X[:5])

print("\nFirst 5 y:")
print(y[:5])


print("\nTotal observations:", len(y))
print("No crash:", np.sum(y == 0))
print("Crash:", np.sum(y == 1))

print("Crash percentage:", np.mean(y) * 100)