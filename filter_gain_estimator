import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === PARAMETERS ===
dt = 1.0
Q = 0.1               # system (process) noise: model uncertainty in gain
R = 0.20               # measurement noise variance
initial_gain_estimate = 0.5
true_gain = 1.2
n = 20                 # number of time steps

# === SIMULATED MEASUREMENTS ===

# === SIMULATED INPUTS ===
u = np.random.choice([-1, 1], size=n)

np.random.seed(42)
z = [true_gain + np.random.normal(0, np.sqrt(R)) for _ in range(n)]         # random sequence of length n consisting of only -1 and 1
measurement_available = [False] + [u[k] != u[k-1] for k in range(1, n)]     # new measurement is only taken when the input changes
z = [z[k] if measurement_available[k] else 0 for k in range(len(z))]        # z[k] zero when there's no measurement available

# === FILTER INITIALIZATION ===
x_est = np.array([initial_gain_estimate])  # gain only
P = np.array([[1.0]])  # initial uncertainty
estimates_gain = []
P_trace = []

# === KALMAN FILTER LOOP (1D state) ===
for k in range(n):
    # Predict step
    x_pred = x_est.copy()
    P = P + Q  # uncertainty increases due to process noise

    if measurement_available[k]:
        # Measurement update
        y = z[k] - x_pred
        S = P + R
        K = P / S
        x_est = x_pred + K * y
        P = (1 - K) * P
    else:
        # No measurement update
        x_est = x_pred

    estimates_gain.append(float(x_est))
    P_trace.append(float(P))

# === PLOTTING ===
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axs[0].plot([true_gain] * n, label="True Gain")
axs[0].plot(z, label="Measured Gain (z)", linestyle="dotted")
axs[0].plot(estimates_gain, label="Estimated Gain", linestyle="--")
axs[0].scatter(
    [i for i, flag in enumerate(measurement_available) if not flag],
    [estimates_gain[i] for i, flag in enumerate(measurement_available) if not flag],
    color='red', label="Missing Measurement", marker='x'
)
axs[0].set_ylabel("Gain")
axs[0].legend()
axs[0].set_title("1D Kalman Filter: Gain Estimation with Missing Measurements")

axs[1].plot(P_trace, label="Uncertainty (P)", color='orange')
axs[1].set_ylabel("P (Variance)")
axs[1].set_xlabel("Time Step")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

# === TABLE OUTPUT ===
df = pd.DataFrame({
    "Time Step": list(range(n)),
    "Measurement z[k]": z,
    "Measurement Available": measurement_available,
    "Estimated Gain": estimates_gain,
    "Uncertainty P": P_trace
})
df.head(10)

n = len(u)

# Simulate true position
true_pos = [0]
for i in range(n):
    new_pos = (1-leak)*true_pos[-1] + true_gain * u[i] * dt
    true_pos.append(new_pos)

# Simulate noisy measurements of gain
z = [true_gain + np.random.normal(0, np.sqrt(R)) for _ in range(n)]

# === FILTER INITIALIZATION ===
x_est = np.array([0.0, initial_gain_estimate])  # [position, gain]
P = np.eye(2)
estimates_pos = []
estimates_gain = []

# Measurement matrix: we observe gain (x2), not position
H = np.array([[0, 1]])

# === EKF LOOP ===
for k in range(n):
    # Predict
    x_pred = np.array([
    (1 - leak) * x_est[0] + x_est[1] * u[k] * dt,
    x_est[1]
    ])
    F = np.array([
    [1 - leak, u[k] * dt],
    [0, 1]
    ])
    P = F @ P @ F.T + Q

    # Measurement update (observe gain)
    y = z[k] - H @ x_pred
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x_est = x_pred + (K.flatten() * y)
    P = (np.eye(2) - K @ H) @ P

    estimates_pos.append(x_est[0])
    estimates_gain.append(x_est[1])

# === PLOTTING ===
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(true_pos[1:], label="True Position")
axs[0].plot(estimates_pos, label="Estimated Position", linestyle="--")
axs[0].set_title("Position Estimate (Only Gain Measured)")
axs[0].set_xlabel("Time Step")
axs[0].set_ylabel("Position")
axs[0].legend()
axs[0].grid(True)

axs[1].plot([true_gain]*n, label="True Gain")
axs[1].plot(z, label="Measured Gain", linestyle="dotted")
axs[1].plot(estimates_gain, label="Estimated Gain", linestyle="--")
axs[1].set_title("Gain Estimate")
axs[1].set_xlabel("Time Step")
axs[1].set_ylabel("Gain")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

# === DATA TABLE OUTPUT ===
df = pd.DataFrame({
    "Time Step": list(range(n)),
    "Control Input u[k]": u,
    "Measured Gain z[k]": z,
    "Estimated Position": estimates_pos,
    "Estimated Gain": estimates_gain,
    "True Position": true_pos[1:]
})
print(df)

