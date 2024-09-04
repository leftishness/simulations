import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Constants and parameters
L = 10.0  # Length of the spatial domain
dx = 0.1  # Spatial step size
dt = 0.01  # Time step size
T = 10.0  # Total simulation time
sigma = 1.0  # Width of the Gaussian
timesteps = int(T / dt)  # Number of time steps
alpha, beta, gamma = 0.1, 0.1, 0.1  # Coupling constants
sigma_eta = 0.001  # Standard deviation for quantum noise
epsilon = 1e-8  # Small positive value to avoid division by zero

def create_spatial_grid(L, dx):
    """Create a 2D spatial grid."""
    x = np.arange(-L / 2, L / 2, dx)
    y = np.arange(-L / 2, L / 2, dx)
    return np.meshgrid(x, y)

def initial_self_entanglement_2d(X, Y, sigma):
    """Initialize the self-entanglement distribution as a Gaussian."""
    return np.exp(-((X - 0.0) ** 2 + (Y - 0.0) ** 2) / (2 * sigma ** 2))

def initial_quantum_metric_2d(X, Y, sigma):
    """Initialize the quantum metric tensor components with slight perturbations."""
    g_tt = 1.0 - 0.05 * np.exp(-((X) ** 2 + (Y) ** 2) / (2 * sigma ** 2))
    g_xx = 1.0 + 0.05 * np.exp(-((X) ** 2 + (Y) ** 2) / (2 * sigma ** 2))
    g_yy = 1.0 + 0.05 * np.exp(-((X) ** 2 + (Y) ** 2) / (2 * sigma ** 2))
    return g_tt, g_xx, g_yy

def calculate_ricci_scalar_2d(g_tt, g_xx, g_yy, dx):
    """Calculate the Ricci scalar from the metric tensor components."""
    g = g_tt * g_xx * g_yy
    sqrt_g = np.sqrt(np.maximum(g, epsilon))

    d_gtt_dx = np.gradient(g_tt, dx, axis=0)
    d_gtt_dy = np.gradient(g_tt, dx, axis=1)

    Gamma_x_tt = 0.5 * g_xx * d_gtt_dx
    Gamma_y_tt = 0.5 * g_yy * d_gtt_dy
    Ricci_scalar = -(np.gradient(sqrt_g * Gamma_x_tt, dx, axis=0) +
                     np.gradient(sqrt_g * Gamma_y_tt, dx, axis=1)) / sqrt_g

    return Ricci_scalar

def evolve_metric_2d(g_tt, g_xx, g_yy, S, dt, alpha, beta, gamma, sigma_eta, dx):
    """Evolve the metric tensor components over time."""
    quantum_correction_tt = np.random.normal(0, sigma_eta, size=g_tt.shape)
    quantum_correction_xx = np.random.normal(0, sigma_eta, size=g_xx.shape)
    quantum_correction_yy = np.random.normal(0, sigma_eta, size=g_yy.shape)

    dg_tt_dt = -alpha * (np.gradient(S, dx, axis=0) + np.gradient(S, dx, axis=1)) + quantum_correction_tt
    dg_xx_dt = beta * np.gradient(S, dx, axis=0) + quantum_correction_xx
    dg_yy_dt = gamma * np.gradient(S, dx, axis=1) + quantum_correction_yy

    g_tt += dg_tt_dt * dt
    g_xx += dg_xx_dt * dt
    g_yy += dg_yy_dt * dt

    # Ensure metric components remain positive
    g_tt = np.maximum(g_tt, epsilon)
    g_xx = np.maximum(g_xx, epsilon)
    g_yy = np.maximum(g_yy, epsilon)

    return g_tt, g_xx, g_yy

def adaptive_mesh_refinement(field, threshold):
    """Identify regions with significant gradients for adaptive mesh refinement."""
    gradients = np.gradient(field)
    magnitude = np.sqrt(gradients[0] ** 2 + gradients[1] ** 2)
    return magnitude > threshold

def compute_entanglement_entropy(S):
    """Compute a 2D entropy-like measure for entanglement."""
    return -S * np.log2(S + epsilon)

def visualize_causal_structure(g_tt, g_xx, g_yy):
    """Visualize the causal structure by computing the light cone directions."""
    c = 1  # Speed of light
    dt = np.sqrt(g_tt / (g_xx * c ** 2))
    dx = np.sqrt(g_xx) * c * dt
    dy = np.sqrt(g_yy) * c * dt
    return dx, dy

def run_simulation(X, Y, timesteps, S, g_tt, g_xx, g_yy, dt, alpha, beta, gamma, sigma_eta, dx):
    """Run the simulation over the specified number of timesteps."""
    for t in tqdm(range(timesteps), desc="Enhanced Simulation Progress"):
        g_tt, g_xx, g_yy = evolve_metric_2d(g_tt, g_xx, g_yy, S, dt, alpha, beta, gamma, sigma_eta, dx)
        Ricci_scalar = calculate_ricci_scalar_2d(g_tt, g_xx, g_yy, dx)

        refine_mask = adaptive_mesh_refinement(Ricci_scalar, threshold=0.1)
        ent_entropy = compute_entanglement_entropy(S)
        dx_cone, dy_cone = visualize_causal_structure(g_tt, g_xx, g_yy)

        if t == timesteps - 1:  # Only plot at the final timestep
            visualize_results(X, Y, g_tt, g_xx, g_yy, Ricci_scalar, ent_entropy, dx_cone, dy_cone)

def visualize_results(X, Y, g_tt, g_xx, g_yy, Ricci_scalar, ent_entropy, dx_cone, dy_cone):
    """Visualize the results of the simulation."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))

    axs[0, 0].imshow(g_tt, extent=[-L / 2, L / 2, -L / 2, L / 2], origin='lower')
    axs[0, 0].set_title("g_tt Evolution")
    axs[0, 1].imshow(Ricci_scalar, extent=[-L / 2, L / 2, -L / 2, L / 2], origin='lower')
    axs[0, 1].set_title("Ricci Scalar Evolution")
    axs[1, 0].imshow(ent_entropy, extent=[-L / 2, L / 2, -L / 2, L / 2], origin='lower')
    axs[1, 0].set_title("Entanglement Entropy")
    axs[1, 1].quiver(X[::5, ::5], Y[::5, ::5], dx_cone[::5, ::5], dy_cone[::5, ::5])
    axs[1, 1].set_title("Causal Structure")

    for ax in axs.flatten():
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()
    plt.show()

def main():
    # Create spatial grid
    X, Y = create_spatial_grid(L, dx)

    # Initialize fields
    S = initial_self_entanglement_2d(X, Y, sigma)
    g_tt, g_xx, g_yy = initial_quantum_metric_2d(X, Y, sigma)

    # Run the simulation
    run_simulation(X, Y, timesteps, S, g_tt, g_xx, g_yy, dt, alpha, beta, gamma, sigma_eta, dx)

if __name__ == "__main__":
    main()
