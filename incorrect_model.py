# Repeated with data generated from a constant‑acceleration system (model mismatch)
# to show the normalized innovations deviate from N(0,1).

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

# Timestep between two consecutive steps in simulation
dT = 1
# Number of steps in simulation
STEPS = 1000
# Measurement noise covariance
R = 3
# Process noise covariance
Q = 1.0 * np.array([ [ (1/3)*dT**3, (1/2)*dT**2 ], [ (1/2)*dT**2, dT] ])
# State evolution matrix
F = np.array([ [1, dT], [0, 1] ])
# Observation matrix
H = np.array([ [1, 0] ])
# Measurement noise source
v = lambda : np.random.normal(0, np.sqrt(R))

# State evolution matrix for generator
Fg = np.array([ [1, dT, 0.5*dT**2], [0, 1, dT], [0, 0, 1] ])
# Observation matrix for generator
Hg = np.array([ [1, 0, 0] ])
# Acceleration (generator)
Ag = 0.8



if __name__ == '__main__':
    # Initial state
    x0 = np.array([5, 2, Ag]).T
    # True state (hidden)
    x = np.zeros([STEPS, 3]).T
    x[:, 0] = x0
    # Measurement
    y = np.zeros([STEPS, 1]).T
    y[:, 0] = Hg @ x0 + v()
    # Estimated state
    z = np.zeros([STEPS, 2]).T
    z[:, 0] = np.array([5, 2]) # TInitial estimate
    # Estimate uncertainty
    P = np.zeros([2, 2, STEPS])
    P[:, :, 0] = np.array([[0, 0], [0, 0]]) # Initial estimate uncertainty
    # Normalized innovation
    nS = np.zeros([STEPS - 1, 1]) # One less datapoint than STEPS

    for i in range(1, STEPS):
        x[:, i] = Fg @ x[:, i-1]
        y[:, i] = Hg @ x[:, i] + v()

        # Kalman filter
        # -- Prediciton
        x_pred = F @ z[:, i-1]
        P_pred = F @ P[:, :, i-1] @ F.T + Q

        # -- Innovation
        y_pred = H @ x_pred
        innov = y[:, i] - y_pred
        S = H @ P_pred @ H.T + R
        # Store normalized innovation
        # Use i-1 because there is STEPS - 1 samples of it
        # this prevents removing first element later
        nS[i-1] = innov / np.sqrt(S)

        # -- Kalman Gain
        K = P_pred @ H.T @ np.linalg.inv(S)

        # -- Update
        z[:, i] = x_pred + K @ innov
        I = np.eye(2)
        P[:, :, i-1] = (I - K @ H) @ P_pred

    t = range(STEPS)
    print(x)
    print(y)
    print(z)

    plt.plot(t, x[0, :], label='State')
    plt.plot(t, y[0, :], label = 'Measurement')
    plt.plot(t, z[0, :], label = 'Estimate')
    plt.xlabel("Time")
    plt.ylabel("Distance")
    plt.title(f"Filter results N={STEPS} dT={dT}")
    plt.legend(loc="upper left")
    plt.show()

    a = np.linspace(-4, 4, 400)
    pdf = norm.pdf(a, loc=0, scale=1)
    plt.figure(figsize=(8,4))
    plt.hist(nS, bins=40, density=True, alpha=0.6, color='steelblue', label='Normalized innovations')
    plt.plot(a, pdf, 'r-', linewidth=2, label='N(0,1)')
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Normalized Innovation Distribution\nAccelerated object / Constant velocity model")
    plt.legend()
    plt.show()
