import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from qiskit import QuantumCircuit
from qiskit_aer import StatevectorSimulator
from qiskit.quantum_info import Statevector, Operator
from scipy.linalg import expm

# Constants and parameters for 1+1 dimensions
dx = 0.05  # Spatial step size for higher resolution along the 1D axis
dt = 0.005  # Time step size for higher temporal resolution
L = 10.0  # Length of the spatial domain
T = 10.0  # Total simulation time
epsilon = 1e-8  # Small value to prevent division by zero or other numerical instabilities


def initialize_grid(L, dx):
    """
    Initialize the spatial grid.

    Parameters:
    - L (float): Length of the spatial domain.
    - dx (float): Spatial step size.

    Returns:
    - np.array: Array representing the spatial grid points.
    """
    return np.arange(-L / 2, L / 2, dx)


def initial_entanglement(x, form='gaussian'):
    """
    Initialize the entanglement distribution along the 1D axis.

    Parameters:
    - x (np.array): Spatial grid points.
    - form (str): Form of the initial entanglement distribution ('gaussian' or 'double_gaussian').

    Returns:
    - np.array: Initial entanglement distribution.
    """
    if form == 'gaussian':
        sigma = 1.0  # Standard deviation of the Gaussian
        return np.exp(-x ** 2 / (2 * sigma ** 2))
    elif form == 'double_gaussian':
        sigma = 0.5  # Standard deviation for the double Gaussian
        return np.exp(-x ** 2 / (2 * sigma ** 2)) + np.exp(-(x - 2) ** 2 / (2 * sigma ** 2))
    else:
        raise ValueError("Unknown initial entanglement form")


def quantum_metric_from_entanglement(S, M):
    """
    Calculate the initial quantum metric tensor components from the entanglement distribution.

    Parameters:
    - S (np.array): Entanglement distribution.
    - M (float): Entanglement monogamy measure.

    Returns:
    - tuple: Components of the quantum metric tensor (g_tt, g_xx).
    """
    g_tt = 1.0 - 0.05 * S * M  # Temporal component
    g_xx = 1.0 - 0.05 * S * M * (1 + 0.05 * np.sin(S))  # Spatial component with slight anisotropy
    return g_tt, g_xx


def evolve_entanglement(S, t, interaction_strength, noise_scale=0.0):
    """
    Evolve the entanglement distribution over time with an optional noise component.

    Parameters:
    - S (np.array): Current entanglement distribution.
    - t (float): Current time.
    - interaction_strength (float): Strength of the interaction affecting entanglement.
    - noise_scale (float): Scale of the noise applied to the entanglement distribution.

    Returns:
    - np.array: Updated entanglement distribution.
    """
    noise = noise_scale * np.random.normal(size=S.shape)  # Generate noise
    return S * np.exp(-interaction_strength * t) * (1 + 0.01 * np.cos(2 * np.pi * t)) + noise


def calculate_monogamy(S):
    """
    Calculate the entanglement monogamy measure.

    Parameters:
    - S (np.array): Entanglement distribution.

    Returns:
    - float: Monogamy measure reflecting the quantum constraints on entanglement distribution.
    """
    return 1.0 / (1.0 + np.sum(S ** 2))


def calculate_ricci_scalar_1d(g_tt, g_xx, dx):
    """
    Calculate the Ricci scalar in 1+1 dimensions from the metric components.

    Parameters:
    - g_tt (np.array): Temporal component of the quantum metric tensor.
    - g_xx (np.array): Spatial component of the quantum metric tensor.
    - dx (float): Spatial step size.

    Returns:
    - np.array: Ricci scalar along the 1D axis.
    """
    g = g_tt * g_xx  # Determinant of the metric tensor
    sqrt_g = np.sqrt(np.maximum(g, epsilon))  # Ensure non-negative determinant

    # Gradients for the Christoffel symbols (simplified for 1D)
    d_gtt_dx = np.gradient(g_tt, dx)
    d_gxx_dx = np.gradient(g_xx, dx)

    # Simplified Ricci scalar calculation
    Ricci_scalar = - (np.gradient(sqrt_g * d_gtt_dx, dx) + np.gradient(sqrt_g * d_gxx_dx, dx)) / sqrt_g

    return Ricci_scalar


def update_metric_tensor(clock_state, g_tt, g_xx, S, M, t, interaction_strength):
    """
    Update the metric tensor based on the clock-system interaction and the current entanglement distribution.

    Parameters:
    - clock_state (Statevector): Current state of the quantum clock.
    - g_tt (np.array): Temporal component of the metric tensor.
    - g_xx (np.array): Spatial component of the metric tensor.
    - S (np.array): Entanglement distribution.
    - M (float): Monogamy measure.
    - t (float): Current time.
    - interaction_strength (float): Strength of the interaction between the clock and system.

    Returns:
    - tuple: Updated components of the quantum metric tensor (g_tt, g_xx).
    """
    rho = clock_state.data  # Get the quantum state data (amplitudes)
    probabilities = np.abs(rho) ** 2  # Calculate probabilities from amplitudes

    # Simplified linear update based on clock states and monogamy measure
    g_tt += interaction_strength * (probabilities[0] - probabilities[1]) * S * M
    g_xx += interaction_strength * (probabilities[0] - probabilities[1]) * S * M

    return g_tt, g_xx


def run_simulation():
    """
    Run the main simulation loop with the model.

    This function sets up the initial conditions, evolves the system over time,
    and computes the Ricci scalar for each time step. Results are plotted at the end.
    """
    x = initialize_grid(L, dx)  # Set up the spatial grid
    S = initial_entanglement(x, form='gaussian')  # Initialize entanglement with a Gaussian distribution
    M = calculate_monogamy(S)  # Compute initial monogamy measure
    g_tt, g_xx = quantum_metric_from_entanglement(S, M)  # Compute initial metric tensor components

    interaction_strength = 0.1  # Set the strength of interaction between the clock and the system

    # Set up the quantum simulator for the clock system
    simulator = StatevectorSimulator()
    clock_state = Statevector.from_label('0')  # Initialize the clock in state |0>

    # Define the Hamiltonian for the clock (using Pauli-Z operator)
    H_clock = np.array([[1, 0], [0, -1]])
    time_evolution_operator = expm(-1j * H_clock * dt)  # Time evolution operator for one time step

    ricci_scalars = []  # List to store Ricci scalar at each time step

    for t in tqdm(range(int(T / dt)), desc="PaW Simulation Progress"):
        clock_state = clock_state.evolve(Operator(time_evolution_operator))  # Evolve the clock state

        S = evolve_entanglement(S, t * dt, interaction_strength, noise_scale=0.0)  # Evolve entanglement without noise
        M = calculate_monogamy(S)  # Update monogamy measure
        g_tt, g_xx = update_metric_tensor(clock_state, g_tt, g_xx, S, M, t * dt,
                                          interaction_strength)  # Update metric tensor

        Ricci_scalar = calculate_ricci_scalar_1d(g_tt, g_xx, dx)  # Calculate Ricci scalar
        max_val = np.max(np.abs(Ricci_scalar))
        if max_val > 0:
            Ricci_scalar /= max_val  # Normalize Ricci scalar for consistent visualization
        ricci_scalars.append(Ricci_scalar)  # Store the normalized Ricci scalar

    plot_results(x, ricci_scalars)  # Plot the results


def plot_results(x, ricci_scalars):
    """
    Plot the simulation results, specifically the evolution of the Ricci scalar.

    Parameters:
    - x (np.array): Spatial grid points.
    - ricci_scalars (list of np.array): List of Ricci scalar values over time.
    """
    plt.figure(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(ricci_scalars[::100])))  # Generate colors for different time steps

    for i, (Ricci_scalar, color) in enumerate(zip(ricci_scalars[::100], colors)):
        plt.plot(x, Ricci_scalar, color=color, label=f"Time: {i * dt * 100:.2f}")  # Plot each time step

    plt.legend()
    plt.xlabel('Position')
    plt.ylabel('Normalized Ricci Scalar')
    plt.title('Normalized Ricci Scalar Evolution')
    plt.show()


if __name__ == "__main__":
    run_simulation()
