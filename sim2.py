import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Constants and parameters for 3+1 dimensions
dx = 0.1  # Spatial step size
dy = 0.1  # Spatial step size
dz = 0.1  # Spatial step size
dt = 0.01  # Time step size
L = 10.0  # Length of the spatial domain
T = 10.0  # Total simulation time
epsilon = 1e-8  # Small positive value to avoid sqrt of negative or zero

x = np.arange(-L / 2, L / 2, dx)  # Spatial grid for x
y = np.arange(-L / 2, L / 2, dy)  # Spatial grid for y
z = np.arange(-L / 2, L / 2, dz)  # Spatial grid for z
X, Y, Z = np.meshgrid(x, y, z)

timesteps = int(T / dt)  # Number of time steps


# Initial conditions for self-entanglement tensor (energy density dependent)
def initial_self_entanglement_3d(X, Y, Z):
    return np.exp(-X ** 2 - Y ** 2 - Z ** 2) + 0.5 * np.exp(-(X - 2) ** 2 - Y ** 2 - Z ** 2)  # Double Gaussian in 3D


# Initial conditions for the quantum metric tensor in 3D
def initial_quantum_metric_3d(X, Y, Z):
    g_tt = 1.0 - 0.1 * np.exp(-X ** 2 - Y ** 2 - Z ** 2)  # Small perturbation to flat space
    g_xx = 1.0 + 0.1 * np.exp(-X ** 2 - Y ** 2 - Z ** 2)
    g_yy = 1.0 + 0.1 * np.exp(-X ** 2 - Y ** 2 - Z ** 2)
    g_zz = 1.0 + 0.1 * np.exp(-X ** 2 - Y ** 2 - Z ** 2)
    return g_tt, g_xx, g_yy, g_zz


# Calculate the Ricci scalar in 3+1 dimensions
def calculate_ricci_scalar_3d(g_tt, g_xx, g_yy, g_zz, dx, dy, dz):
    g = g_tt * g_xx * g_yy * g_zz  # Determinant of the metric
    sqrt_g = np.sqrt(np.maximum(g, epsilon))  # Ensure g is positive

    # Christoffel symbols (simplified approximation)
    d_gtt_dx = np.gradient(g_tt, dx, axis=0)
    d_gtt_dy = np.gradient(g_tt, dy, axis=1)
    d_gtt_dz = np.gradient(g_tt, dz, axis=2)
    d_gxx_dx = np.gradient(g_xx, dx, axis=0)
    d_gyy_dy = np.gradient(g_yy, dy, axis=1)
    d_gzz_dz = np.gradient(g_zz, dz, axis=2)

    # Simplified form of Ricci scalar in 3+1 dimensions (assuming diagonal metric)
    Gamma_x_tt = 0.5 * g_xx * d_gtt_dx
    Gamma_y_tt = 0.5 * g_yy * d_gtt_dy
    Gamma_z_tt = 0.5 * g_zz * d_gtt_dz
    Ricci_scalar = - (np.gradient(sqrt_g * Gamma_x_tt, dx, axis=0) +
                      np.gradient(sqrt_g * Gamma_y_tt, dy, axis=1) +
                      np.gradient(sqrt_g * Gamma_z_tt, dz, axis=2)) / sqrt_g

    return Ricci_scalar


# Time evolution of the quantum metric tensor in 3D with tunable parameters
def evolve_metric_3d(g_tt, g_xx, g_yy, g_zz, E, dt, alpha=0.1, beta=0.1, gamma=0.1, delta=0.1,
                     quantum_correction_scale=0.001):
    quantum_correction_tt = np.random.normal(0, quantum_correction_scale, size=g_tt.shape)
    quantum_correction_xx = np.random.normal(0, quantum_correction_scale, size=g_xx.shape)
    quantum_correction_yy = np.random.normal(0, quantum_correction_scale, size=g_yy.shape)
    quantum_correction_zz = np.random.normal(0, quantum_correction_scale, size=g_zz.shape)

    dg_tt_dt = -alpha * (np.gradient(E, dx, axis=0) + np.gradient(E, dy, axis=1) + np.gradient(E, dz,
                                                                                               axis=2)) + quantum_correction_tt
    dg_xx_dt = beta * np.gradient(E, dx, axis=0) + quantum_correction_xx
    dg_yy_dt = gamma * np.gradient(E, dy, axis=1) + quantum_correction_yy
    dg_zz_dt = delta * np.gradient(E, dz, axis=2) + quantum_correction_zz

    g_tt += dg_tt_dt * dt
    g_xx += dg_xx_dt * dt
    g_yy += dg_yy_dt * dt
    g_zz += dg_zz_dt * dt

    # Ensure metric components stay non-negative
    g_tt = np.maximum(g_tt, epsilon)
    g_xx = np.maximum(g_xx, epsilon)
    g_yy = np.maximum(g_yy, epsilon)
    g_zz = np.maximum(g_zz, epsilon)

    return g_tt, g_xx, g_yy, g_zz


# Feedback mechanism for self-entanglement tensor in 3D
def update_self_entanglement_3d(g_tt, g_xx, g_yy, g_zz):
    return np.exp(-X ** 2 - Y ** 2 - Z ** 2) * (g_tt - g_xx - g_yy - g_zz)


# General Relativity Prediction: Ricci scalar for a spherically symmetric mass distribution
def gr_ricci_scalar_3d(X, Y, Z, M=1):
    r_squared = X ** 2 + Y ** 2 + Z ** 2
    r = np.sqrt(r_squared + epsilon)  # Avoid division by zero
    ricci_scalar_gr = (2 * M / r_squared) * np.exp(-r_squared)  # Simplified GR prediction for a Gaussian mass
    return ricci_scalar_gr


# Setup for subplots
fig, axs = plt.subplots(3, 3, figsize=(18, 12))
axs[0, 0].set_title("Evolution of g_tt Over Time (x=0)")
axs[0, 1].set_title("Evolution of g_tt Over Time (y=0)")
axs[0, 2].set_title("Evolution of g_tt Over Time (z=0)")
axs[1, 0].set_title("Evolution of g_xx Over Time (x=0)")
axs[1, 1].set_title("Evolution of g_xx Over Time (y=0)")
axs[1, 2].set_title("Evolution of g_xx Over Time (z=0)")
axs[2, 0].set_title("Ricci Scalar Comparison (x=0)")
axs[2, 1].set_title("Ricci Scalar Comparison (y=0)")
axs[2, 2].set_title("Ricci Scalar Comparison (z=0)")

for i in range(3):
    for j in range(3):
        axs[i, j].set_xlabel("x" if j == 0 else ("y" if j == 1 else "z"))
        axs[i, j].set_ylabel("g_tt" if i == 0 else ("g_xx" if i == 1 else "Ricci Scalar"))

# Initialize fields
E = initial_self_entanglement_3d(X, Y, Z)  # Complex initial condition for self-entanglement tensor
g_tt, g_xx, g_yy, g_zz = initial_quantum_metric_3d(X, Y, Z)  # Quantum metric tensor components

# Run the simulation with feedback, quantum corrections, and adjusted parameters in 3D
alpha = 0.1
beta = 0.1
gamma = 0.1
delta = 0.1
quantum_correction_scale = 0.001

# Use tqdm to add a progress bar
for t in tqdm(range(timesteps), desc="Simulation Progress"):
    g_tt, g_xx, g_yy, g_zz = evolve_metric_3d(g_tt, g_xx, g_yy, g_zz, E, dt, alpha, beta, gamma, delta,
                                              quantum_correction_scale)
    E = update_self_entanglement_3d(g_tt, g_xx, g_yy, g_zz)  # Update E based on the evolving metric

    if t % 100 == 0:  # Plot every 100 time steps
        Ricci_scalar_self_entanglement = calculate_ricci_scalar_3d(g_tt, g_xx, g_yy, g_zz, dx, dy, dz)
        Ricci_scalar_gr = gr_ricci_scalar_3d(X, Y, Z)

        # Normalize the self-entanglement Ricci scalar to match the GR prediction's scale
        normalization_factor = np.max(np.abs(Ricci_scalar_gr))
        Ricci_scalar_self_entanglement_normalized = Ricci_scalar_self_entanglement / np.max(
            np.abs(Ricci_scalar_self_entanglement)) * normalization_factor

        # Plot along all slices: x=0, y=0, z=0
        slice_idx_x = X.shape[0] // 2
        slice_idx_y = Y.shape[1] // 2
        slice_idx_z = Z.shape[2] // 2

        axs[0, 0].plot(y, g_tt[slice_idx_x, :, slice_idx_z], label=f"Time: {t * dt:.2f}")
        axs[0, 1].plot(x, g_tt[:, slice_idx_y, slice_idx_z], label=f"Time: {t * dt:.2f}")
        axs[0, 2].plot(x, g_tt[:, slice_idx_y, slice_idx_z], label=f"Time: {t * dt:.2f}")

        axs[1, 0].plot(y, g_xx[slice_idx_x, :, slice_idx_z], label=f"Time: {t * dt:.2f}")
        axs[1, 1].plot(x, g_xx[:, slice_idx_y, slice_idx_z], label=f"Time: {t * dt:.2f}")
        axs[1, 2].plot(x, g_xx[:, slice_idx_y, slice_idx_z], label=f"Time: {t * dt:.2f}")

        axs[2, 0].plot(y, Ricci_scalar_self_entanglement_normalized[slice_idx_x, :, slice_idx_z],
                       label=f"QE {t * dt:.2f}")
        axs[2, 0].plot(y, Ricci_scalar_gr[slice_idx_x, :, slice_idx_z], label="GR", linestyle='--')

        axs[2, 1].plot(x, Ricci_scalar_self_entanglement_normalized[:, slice_idx_y, slice_idx_z],
                       label=f"QE {t * dt:.2f}")
        axs[2, 1].plot(x, Ricci_scalar_gr[:, slice_idx_y, slice_idx_z], label="GR", linestyle='--')

        axs[2, 2].plot(x, Ricci_scalar_self_entanglement_normalized[:, slice_idx_y, slice_idx_z],
                       label=f"QE {t * dt:.2f}")
        axs[2, 2].plot(x, Ricci_scalar_gr[:, slice_idx_y, slice_idx_z], label="GR", linestyle='--')

# Finalize and show plots
for i in range(3):
    for j in range(3):
        axs[i, j].legend(loc='upper right')
plt.tight_layout()
plt.show()
