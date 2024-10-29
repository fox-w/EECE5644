import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
sigma_x = sigma_y = 0.25
prior_cov = np.diag([sigma_x**2, sigma_y**2])
sigma_measure = 0.3
x_true, y_true = np.random.uniform(-1, 1, 2)
true_position = np.array([x_true, y_true])

# Function to generate K landmarks on unit circle
def generate_landmarks(K):
    angles = np.linspace(0, 2*np.pi, K, endpoint=False)
    landmarks = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    return landmarks

# Function to generate range measurements
def generate_measurements(landmarks, true_pos, sigma):
    K = landmarks.shape[0]
    measurements = []
    for i in range(K):
        d = np.linalg.norm(true_pos - landmarks[i])
        r = d + np.random.normal(0, sigma)
        while r < 0:
            r = d + np.random.normal(0, sigma)
        measurements.append(r)
    return np.array(measurements)

# Define the objective function (negative log-posterior)
def objective(pos, landmarks, measurements):
    x, y = pos
    prior = (x**2)/(2*sigma_x**2) + (y**2)/(2*sigma_y**2)
    likelihood = 0
    for i in range(len(measurements)):
        d = np.sqrt((x - landmarks[i,0])**2 + (y - landmarks[i,1])**2)
        likelihood += ((measurements[i] - d)**2)/(2*sigma_measure**2)
    return prior + likelihood

# Grid for plotting
grid_size = 200
x = np.linspace(-2, 2, grid_size)
y = np.linspace(-2, 2, grid_size)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Iterate over K = 1 to 4
for K in range(1, 5):
    landmarks = generate_landmarks(K)
    measurements = generate_measurements(landmarks, true_position, sigma_measure)
    
    # Compute objective over grid
    for i in range(grid_size):
        for j in range(grid_size):
            Z[j,i] = objective([X[j,i], Y[j,i]], landmarks, measurements)
    
    # Find MAP estimate
    res = minimize(objective, [0,0], args=(landmarks, measurements))
    map_estimate = res.x
    
    # Plot contours
    plt.figure(figsize=(6,6))
    CS = plt.contour(X, Y, Z, levels=30, cmap='viridis')
    plt.clabel(CS, inline=1, fontsize=10)
    plt.plot(true_position[0], true_position[1], 'r+', markersize=12, label='True Position')
    plt.plot(map_estimate[0], map_estimate[1], 'bx', markersize=12, label='MAP Estimate')
    plt.plot(landmarks[:,0], landmarks[:,1], 'ko', label='Landmarks')
    plt.title(f'MAP Estimation Contours for K={K}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.show()
