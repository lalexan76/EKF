# Re-run complete Kalman filter code after execution environment reset

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === PARAMETERS ===
dt = 1.0  # Time step
Q = np.diag([0.1, 0.01])  # Process noise (for position and gain)
R = 0.1  # Measurement noise variance (for gain)
initial_gain_estimate = 0.8
true_gain = 0.7  # True but hidden value
leak = 0.4  # for example

# === SIMULATED INPUTS ===
u = [1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1]  # control inputs
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

