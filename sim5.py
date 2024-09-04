import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

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

def gaussian_entanglement(X, Y, x0=0.0, y0=0.0, sigma=sigma):
    """Create a Gaussian entanglement distribution centered at (x0, y0)."""
    return np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2))

def double_gaussian_entanglement(X, Y, x1=-2.0, y1=0.0, x2=2.0, y2=0.0, sigma=sigma):
    """Create a double Gaussian entanglement distribution centered at (x1, y1) and (x2, y2)."""
    return (np.exp(-((X - x1) ** 2 + (Y - y1) ** 2) / (2 * sigma ** 2)) +
            np.exp(-((X - x2) ** 2 + (Y - y2) ** 2) / (2 * sigma ** 2)))

def uniform_entanglement(X, Y):
    """Create a uniform entanglement distribution."""
    return np.ones_like(X)

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

def evolve_metric_2d(g_tt, g_xx, g_yy, S, dt, alpha, beta, gamma, quantum_noise_scale, dx):
    """Evolve the metric tensor components over time."""
    quantum_correction_tt = np.random.normal(0, quantum_noise_scale, size=g_tt.shape)
    quantum_correction_xx = np.random.normal(0, quantum_noise_scale, size=g_xx.shape)
    quantum_correction_yy = np.random.normal(0, quantum_noise_scale, size=g_yy.shape)

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

def calculate_energy_momentum_tensor(g_tt, g_xx, g_yy, S, dx):
    """Calculate the components of the energy-momentum tensor."""
    T_tt = -(np.gradient(S, dx, axis=0) ** 2 + np.gradient(S, dx, axis=1) ** 2)
    T_xx = np.gradient(S, dx, axis=0) ** 2 - np.gradient(S, dx, axis=1) ** 2
    T_yy = -np.gradient(S, dx, axis=0) ** 2 + np.gradient(S, dx, axis=1) ** 2
    return T_tt, T_xx, T_yy

def visualize_causal_structure(g_tt, g_xx, g_yy):
    """Visualize the causal structure by computing the light cone directions."""
    c = 1  # Speed of light
    dt = np.sqrt(g_tt / (g_xx * c ** 2))
    dx = np.sqrt(g_xx) * c * dt
    dy = np.sqrt(g_yy) * c * dt
    return dx, dy

def detect_singularities(Ricci_scalar, threshold=1e3):
    """Detect potential singularities based on the Ricci scalar."""
    return np.abs(Ricci_scalar) > threshold

def run_simulation(X, Y, entanglement_distributions, timesteps, dt, alpha, beta, gamma, sigma_eta, dx):
    """Run the simulation for different entanglement distributions and visualize the results."""
    fig, axs = plt.subplots(len(entanglement_distributions), 4, figsize=(20, 5 * len(entanglement_distributions)))

    for idx, (label, S) in enumerate(entanglement_distributions.items()):
        # Initialize metric tensor
        g_tt, g_xx, g_yy = initial_quantum_metric_2d(X, Y, sigma)

        # Time evolution of the metric tensor
        for t in tqdm(range(timesteps), desc=f"Simulation Progress ({label})"):
            g_tt, g_xx, g_yy = evolve_metric_2d(g_tt, g_xx, g_yy, S, dt, alpha, beta, gamma, sigma_eta, dx)

        # Calculate final state properties
        Ricci_scalar = calculate_ricci_scalar_2d(g_tt, g_xx, g_yy, dx)
        T_tt, T_xx, T_yy = calculate_energy_momentum_tensor(g_tt, g_xx, g_yy, S, dx)
        singularities = detect_singularities(Ricci_scalar)
        dx_cone, dy_cone = visualize_causal_structure(g_tt, g_xx, g_yy)

        # Smooth the Ricci scalar for better visualization
        Ricci_scalar_smooth = gaussian_filter(Ricci_scalar, sigma=1)

        # Plot the results
        plot_results(axs[idx], label, X, Y, g_tt, Ricci_scalar_smooth, T_tt, dx_cone, dy_cone, singularities)

    plt.tight_layout()
    plt.show()

def plot_results(axs_row, label, X, Y, g_tt, Ricci_scalar_smooth, T_tt, dx_cone, dy_cone, singularities):
    """Plot the simulation results for a single entanglement distribution."""
    axs_row[0].imshow(g_tt, extent=[-L / 2, L / 2, -L / 2, L / 2], origin='lower', cmap='viridis')
    axs_row[0].set_title(f"g_tt Evolution ({label})")

    axs_row[1].imshow(Ricci_scalar_smooth, extent=[-L / 2, L / 2, -L / 2, L / 2], origin='lower', cmap='RdBu_r')
    axs_row[1].set_title(f"Ricci Scalar ({label})")

    axs_row[2].imshow(T_tt, extent=[-L / 2, L / 2, -L / 2, L / 2], origin='lower', cmap='plasma')
    axs_row[2].set_title(f"Energy Density T_tt ({label})")

    axs_row[3].quiver(X[::5, ::5], Y[::5, ::5], dx_cone[::5, ::5], dy_cone[::5, ::5], scale=50)
    axs_row[3].set_title(f"Causal Structure ({label})")
    axs_row[3].set_xlim([-L / 2, L / 2])
    axs_row[3].set_ylim([-L / 2, L / 2])

    # Overlay singularities on Ricci scalar plot
    axs_row[1].contour(X, Y, singularities, levels=[0.5], colors='white', linewidths=0.5)

    for ax in axs_row:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

def calculate_and_print_metrics(X, Y, entanglement_distributions, timesteps, dt, alpha, beta, gamma, sigma_eta, dx):
    """Calculate and print quantitative metrics for each entanglement distribution."""
    for label, S in entanglement_distributions.items():
        g_tt, g_xx, g_yy = initial_quantum_metric_2d(X, Y, sigma)
        for t in range(timesteps):
            g_tt, g_xx, g_yy = evolve_metric_2d(g_tt, g_xx, g_yy, S, dt, alpha, beta, gamma, sigma_eta, dx)

        Ricci_scalar = calculate_ricci_scalar_2d(g_tt, g_xx, g_yy, dx)
        T_tt, _, _ = calculate_energy_momentum_tensor(g_tt, g_xx, g_yy, S, dx)

        print(f"\nMetrics for {label} distribution:")
        print(f"Max Ricci Scalar: {np.max(Ricci_scalar):.2e}")
        print(f"Min Ricci Scalar: {np.min(Ricci_scalar):.2e}")
        print(f"Mean Energy Density: {np.mean(T_tt):.2e}")
        print(f"Max Energy Density: {np.max(T_tt):.2e}")

def main():
    # Create spatial grid
    X, Y = create_spatial_grid(L, dx)

    # Define initial entanglement distributions
    entanglement_distributions = {
        "Gaussian": gaussian_entanglement(X, Y),
        "Double Gaussian": double_gaussian_entanglement(X, Y),
        "Uniform": uniform_entanglement(X, Y)
    }

    # Run the simulation and visualize results
    run_simulation(X, Y, entanglement_distributions, timesteps, dt, alpha, beta, gamma, sigma_eta, dx)

    # Calculate and print quantitative metrics
    calculate_and_print_metrics(X, Y, entanglement_distributions, timesteps, dt, alpha, beta, gamma, sigma_eta, dx)

if __name__ == "__main__":
    main()
