# Kalman Filter Model Mismatch and Innovation Statistics

## Problem Statement
This project explores how the normalized innovations of a Kalman filter behave when the model is correct versus when the model is mismatched. The goal is to demonstrate that the normalized innovation follows a standard normal distribution when model is consistent with (or close to) the real world (data generator), and that normalized innovation deviates from standard normal distribution when model is inconsistent.

To test this, the project assumes that the tracked object moves according to the constant-velocity dynamics. First, in [`correct_model.py`](./correct_model.py) the model used in Kalman filter is the same as the one used to generate data i.e. constant-velocity - we expect to see that normalized innovation follows N(0, 1) in this case.

This is contrasted to [`incorrect_model.py`](./incorrect_model.py) where Kalman filter assumes the same constant-velocity model but the data is generated according to constant-acceleration dynamics.

## Process (Constant-Velocity) Model

In case of correct (constant-velocity) model, state is represented as:

$$ x_t = \begin{bmatrix}
 p_t \\
 v_t
\end{bmatrix} $$

where $x_t$ represents (true) state vector used when generating data, $p_t$ position of the object relative to some arbitrary reference point and $v_t$ velocity of an object at time $`t`$.

In case of incorrect (constant-acceleration) model, state is represented as:

$$ x_t = \begin{bmatrix}
 p_t \\
 v_t \\
 a_t
\end{bmatrix} $$

Measurement:
$$ y_t = p_t $$

and state estimate:

$$ z_t = \begin{bmatrix}
 p_t \\
 v_t
\end{bmatrix} $$


System is modeled as follows:

$$ x_t = F x_{t-1} + w_t \\
(z_t = F z_{t-1} + w_t) \\
y_t = H x_{t} + v_t $$

where $F$ is the state transition matrix, $H$ is observation matrix, $w_t$ is process noise (AWGN) with covariance matrix $Q$ and $w_t$ is measurement noise (AWGN) $v_t$ with variance $R$:

State transition matrix for constant-velocity model is defined as 

$$
    F = \begin{bmatrix}
    1 & \Delta{t} \\
    0 & 1 
    \end{bmatrix}
$$

or in constant-acceleration model (used only in data generation):

$$
    F = \begin{bmatrix}
    1 & \Delta{t} & \frac{1}{2} {\Delta{t}}^2 \\
    0 & 1 & \Delta{t} \\
    0 & 0 & 1 
    \end{bmatrix}
$$

Moreover, measurement noise covariance:

$$
    R = 3
$$

Process noise covariance matrix is modeled as white acceleration noise:

$$
    Q = \sigma{Q} \begin{bmatrix}
        \frac{\Delta{t}^3}{3} & \frac{\Delta{t}^2}{2} \\
        \frac{\Delta{t}^2}{2} & \Delta{t}
    \end{bmatrix}
$$

And finally, state covariance matrix $P$.

## Kalman filter

### Prediction step
$$
    z_{t}^{-} = F z_{t-1} \\
    P_{t}^{-} = F P F^T + Q
$$

### Innovation step
$$
    \widetilde{y_t} = y_t - H z_{t}^{-} \\
    S_t = H P_{t}^{-} H^T + R \\
    \\
    \widetilde{y_{t}}_{norm} = \frac{\widetilde{y_t}}{\sqrt{S_t}}
$$

### Kalman Gain Step
$$
    K_t = P_{t}^{-} H^T S^{-1}
$$

### Update Step
$$
    z_{t} = z_{t}^{-} + K_t \widetilde{y_t} \\
    P_{t} = (I - K_t H) P_{t}^{-}
$$


## Results

This section summarizes the key findings from the simulation.

### Correct Model

![correct_model_position_estimation_graph](./results/correct_model_position_estimation_graph.png)
![correct_model_norm_innov_hist](./results/correct_model_norm_innov_hist.png)

When the filter uses the same model that generated the data (e.g., CA truth + CA filter):
- The histogram of normalized innovations closely matches the standard normal PDF
- The mean is approximately 0
- The variance is approximately 1

This confirms that the Kalman filter is consistent.

### Model Mismatch

![incorrect_model_position_estimation_graph](./results/incorrect_model_position_estimation_graph.png)
![incorrect_model_norm_innov_hist](./results/incorrect_model_norm_innov_hist.png)

When the filter uses the wrong model (e.g., CA truth + CV filter), even though state estimation performance is acceptable, it can be seen through that normalized innovation that:
- The histogram becomes wider and skewed
- The variance is greater than 1
- The mean drifts away from 0

This indicates the filter is inconsistent, and the innovations no longer follow N(0,1).

## Interpretation
A correct model produces innovations that behave like white Gaussian noise.
A mismatched model produces innovations that contain structure, bias, or correlation.
This makes normalized innovations a useful tool for model validation, fault detection, and filter tuning.

## Author
Petar Kaselj

### Acknowledgments
Special thanks to Bojan Vondra for guidance, discussions, and valuable tutoring.
