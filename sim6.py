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
    """Create the initial self-entanglement distribution."""
    return np.exp(-((X) ** 2 + (Y) ** 2) / (2 * sigma ** 2))

def initial_quantum_metric_2d(X, Y, sigma):
    """Initialize the quantum metric tensor components."""
    g_tt = 1.0 - 0.05 * np.exp(-((X) ** 2 + (Y) ** 2) / (2 * sigma ** 2))
    g_xx = 1.0 + 0.05 * np.exp(-((X) ** 2 + (Y) ** 2) / (2 * sigma ** 2))
    g_yy = 1.0 + 0.05 * np.exp(-((X) ** 2 + (Y) ** 2) / (2 * sigma ** 2))
    return g_tt, g_xx, g_yy

def calculate_ricci_scalar_2d(g_tt, g_xx, g_yy, dx, epsilon):
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

def evolve_metric_2d(g_tt, g_xx, g_yy, S, dt, alpha, beta, gamma, quantum_noise_scale, dx, epsilon):
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

def visualize_causal_structure(g_tt, g_xx, g_yy):
    """Visualize the causal structure by computing the light cone directions."""
    c = 1  # Speed of light
    dt = np.sqrt(g_tt / (g_xx * c ** 2))
    dx = np.sqrt(g_xx) * c * dt
    dy = np.sqrt(g_yy) * c * dt
    return dx, dy

def compute_entanglement_entropy(S):
    """Compute a 2D entropy-like measure from the entanglement field."""
    return -S * np.log2(S + 1e-10)

def calculate_energy_momentum_tensor(g_tt, g_xx, g_yy, S, dx):
    """Calculate the components of the energy-momentum tensor."""
    T_tt = -(np.gradient(S, dx, axis=0) ** 2 + np.gradient(S, dx, axis=1) ** 2)
    T_xx = np.gradient(S, dx, axis=0) ** 2 - np.gradient(S, dx, axis=1) ** 2
    T_yy = -np.gradient(S, dx, axis=0) ** 2 + np.gradient(S, dx, axis=1) ** 2
    return T_tt, T_xx, T_yy

def run_simulation(X, Y, timesteps, dt, alpha, beta, gamma, sigma_eta, dx, epsilon):
    """Run the quantum spacetime simulation with and without decoherence."""

    # Prepare to visualize the evolution
    fig1, axs1 = plt.subplots(2, 3, figsize=(18, 12))

    # Run simulation with decoherence
    S = initial_self_entanglement_2d(X, Y, sigma)
    g_tt, g_xx, g_yy = initial_quantum_metric_2d(X, Y, sigma)
    for t in tqdm(range(timesteps), desc="Simulation Progress (Decoherence)"):
        g_tt, g_xx, g_yy = evolve_metric_2d(g_tt, g_xx, g_yy, S, dt, alpha, beta, gamma, sigma_eta, dx, epsilon)
    Ricci_scalar_decoherence = calculate_ricci_scalar_2d(g_tt, g_xx, g_yy, dx, epsilon)
    entanglement_entropy_decoherence = compute_entanglement_entropy(S)
    T_tt_decoherence, _, _ = calculate_energy_momentum_tensor(g_tt, g_xx, g_yy, S, dx)
    dx_cone_decoherence, dy_cone_decoherence = visualize_causal_structure(g_tt, g_xx, g_yy)

    # Run simulation without decoherence
    S = initial_self_entanglement_2d(X, Y, sigma)
    g_tt, g_xx, g_yy = initial_quantum_metric_2d(X, Y, sigma)
    for t in tqdm(range(timesteps), desc="Simulation Progress (No Decoherence)"):
        g_tt, g_xx, g_yy = evolve_metric_2d(g_tt, g_xx, g_yy, S, dt, alpha, beta, gamma, 0, dx, epsilon)
    Ricci_scalar_no_decoherence = calculate_ricci_scalar_2d(g_tt, g_xx, g_yy, dx, epsilon)
    entanglement_entropy_no_decoherence = compute_entanglement_entropy(S)
    T_tt_no_decoherence, _, _ = calculate_energy_momentum_tensor(g_tt, g_xx, g_yy, S, dx)
    dx_cone_no_decoherence, dy_cone_no_decoherence = visualize_causal_structure(g_tt, g_xx, g_yy)

    # Plot g_tt, g_xx, and Ricci Scalar evolution
    plot_evolution(axs1, g_tt, g_xx, Ricci_scalar_decoherence, Ricci_scalar_no_decoherence)
    plt.show()

    # Plot causal structure, entanglement entropy, and energy density
    fig2, axs2 = plt.subplots(2, 3, figsize=(18, 12))
    plot_metrics(axs2, X, Y, dx_cone_decoherence, dy_cone_decoherence, entanglement_entropy_decoherence, T_tt_decoherence,
                 dx_cone_no_decoherence, dy_cone_no_decoherence, entanglement_entropy_no_decoherence, T_tt_no_decoherence)
    plt.show()

def plot_evolution(axs1, g_tt, g_xx, Ricci_scalar_decoherence, Ricci_scalar_no_decoherence):
    """Plot the evolution of g_tt, g_xx, and Ricci Scalar for both decoherence and no decoherence cases."""
    axs1[0, 0].imshow(g_tt, extent=[-L / 2, L / 2, -L / 2, L / 2], origin='lower', cmap='viridis')
    axs1[0, 0].set_title("g_tt Evolution (Decoherence)")
    axs1[0, 1].imshow(g_xx, extent=[-L / 2, L / 2, -L / 2, L / 2], origin='lower', cmap='viridis')
    axs1[0, 1].set_title("g_xx Evolution (Decoherence)")
    axs1[0, 2].imshow(Ricci_scalar_decoherence, extent=[-L / 2, L / 2, -L / 2, L / 2], origin='lower', cmap='RdBu')
    axs1[0, 2].set_title("Ricci Scalar (Decoherence)")

    axs1[1, 0].imshow(g_tt, extent=[-L / 2, L / 2, -L / 2, L / 2], origin='lower', cmap='viridis')
    axs1[1, 0].set_title("g_tt Evolution (No Decoherence)")
    axs1[1, 1].imshow(g_xx, extent=[-L / 2, L / 2, -L / 2, L / 2], origin='lower', cmap='viridis')
    axs1[1, 1].set_title("g_xx Evolution (No Decoherence)")
    axs1[1, 2].imshow(Ricci_scalar_no_decoherence, extent=[-L / 2, L / 2, -L / 2, L / 2], origin='lower', cmap='RdBu')
    axs1[1, 2].set_title("Ricci Scalar (No Decoherence)")

def plot_metrics(axs2, X, Y, dx_cone_decoherence, dy_cone_decoherence, entanglement_entropy_decoherence, T_tt_decoherence,
                 dx_cone_no_decoherence, dy_cone_no_decoherence, entanglement_entropy_no_decoherence, T_tt_no_decoherence):
    """Plot the causal structure, entanglement entropy, and energy density for both decoherence and no decoherence cases."""
    axs2[0, 0].quiver(X[::5, ::5], Y[::5, ::5], dx_cone_decoherence[::5, ::5], dy_cone_decoherence[::5, ::5], scale=50)
    axs2[0, 0].set_title("Causal Structure (Decoherence)")
    axs2[0, 0].set_xlim([-L / 2, L / 2])
    axs2[0, 0].set_ylim([-L / 2, L / 2])

    axs2[0, 1].imshow(entanglement_entropy_decoherence, extent=[-L / 2, L / 2, -L / 2, L / 2], origin='lower', cmap='inferno')
    axs2[0, 1].set_title("Entanglement Entropy (Decoherence)")

    axs2[0, 2].imshow(T_tt_decoherence, extent=[-L / 2, L / 2, -L / 2, L / 2], origin='lower', cmap='plasma')
    axs2[0, 2].set_title("Energy Density T_tt (Decoherence)")

    axs2[1, 0].quiver(X[::5, ::5], Y[::5, ::5], dx_cone_no_decoherence[::5, ::5], dy_cone_no_decoherence[::5, ::5], scale=50)
    axs2[1, 0].set_title("Causal Structure (No Decoherence)")
    axs2[1, 0].set_xlim([-L / 2, L / 2])
    axs2[1, 0].set_ylim([-L / 2, L / 2])

    axs2[1, 1].imshow(entanglement_entropy_no_decoherence, extent=[-L / 2, L / 2, -L / 2, L / 2], origin='lower', cmap='inferno')
    axs2[1, 1].set_title("Entanglement Entropy (No Decoherence)")

    axs2[1, 2].imshow(T_tt_no_decoherence, extent=[-L / 2, L / 2, -L / 2, L / 2], origin='lower', cmap='plasma')
    axs2[1, 2].set_title("Energy Density T_tt (No Decoherence)")

if __name__ == "__main__":
    X, Y = create_spatial_grid(L, dx)
    run_simulation(X, Y, timesteps, dt, alpha, beta, gamma, sigma_eta, dx, epsilon)
