import numpy as np

np.random.seed(42)

N = 1000

# -------------------------
# Generate fake features
# -------------------------

traffic_volume = np.random.randint(200, 2501, N)

average_speed = np.random.uniform(15, 60, N)

rain = np.random.randint(0, 2, N)

night = np.random.randint(0, 2, N)


# -------------------------
# Hidden crash relationship
# -------------------------

z = (
    -6
    + 0.0015 * traffic_volume
    + 0.05 * average_speed
    + 1.0 * rain
    + 0.7 * night
)

crash_probability = 1 / (1 + np.exp(-z))


# -------------------------
# Generate crash / no crash
# -------------------------

crash = np.random.binomial(1, crash_probability)


# -------------------------
# Look at first 10 examples
# -------------------------

for i in range(10):
    print(
        traffic_volume[i],
        round(average_speed[i], 1),
        rain[i],
        night[i],
        round(crash_probability[i], 3),
        crash[i]
    )


# -------------------------
# Save to file so downstream
# scripts don't regenerate it
# -------------------------

data = np.column_stack(
    (traffic_volume, average_speed, rain, night, crash)
)

np.savetxt(
    "data/crash_data.csv",
    data,
    delimiter=",",
    header="traffic_volume,average_speed,rain,night,crash",
    comments="",
    fmt="%.6f"
)

print("\nSaved data/crash_data.csv:", data.shape)