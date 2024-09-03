import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from qiskit import QuantumCircuit
from qiskit_aer import StatevectorSimulator
from qiskit.quantum_info import Statevector, DensityMatrix, Operator
from scipy.linalg import expm

# Constants and parameters for 3+1 dimensions
dx, dy, dz = 0.1, 0.1, 0.1  # Spatial step sizes
dt = 0.01  # Time step size
L = 10.0  # Length of the spatial domain
T = 10.0  # Total simulation time
epsilon = 1e-8  # Small value to prevent division by zero

# Spatial grid
x = np.arange(-L / 2, L / 2, dx)
y = np.arange(-L / 2, L / 2, dy)
z = np.arange(-L / 2, L / 2, dz)
X, Y, Z = np.meshgrid(x, y, z)

timesteps = int(T / dt)  # Number of time steps

# Initialize the quantum clock as a 2-level system (qubit)
qc = QuantumCircuit(1)
clock_state = Statevector.from_label('0')  # Initialize the clock in state |0>

# Pauli matrices for time evolution
H_clock = np.array([[1, 0], [0, -1]])  # Clock Hamiltonian (using sigma_z)
time_evolution_operator = expm(-1j * H_clock * dt)  # U = exp(-iHdt)


# Initial global quantum state setup (initial entanglement)
def initial_entanglement(X, Y, Z):
    return np.exp(-X ** 2 - Y ** 2 - Z ** 2)  # Initial entanglement distribution


# Quantum metric tensor initially based on entanglement
def quantum_metric_from_entanglement(S):
    g_tt = 1.0 - 0.1 * S  # Example mapping for g_tt
    g_xx = 1.0 - 0.1 * S  # Now decrease to potentially produce negative curvature
    g_yy = 1.0 - 0.1 * S
    g_zz = 1.0 - 0.1 * S
    return g_tt, g_xx, g_yy, g_zz


# Entanglement measure evolution
def evolve_entanglement(S, t):
    return S + 0.01 * np.sin(t)  # Example evolution function for entanglement


# Calculate the Ricci scalar in 3+1 dimensions
def calculate_ricci_scalar_3d(g_tt, g_xx, g_yy, g_zz, dx, dy, dz):
    g = g_tt * g_xx * g_yy * g_zz  # Determinant of the metric
    sqrt_g = np.sqrt(np.maximum(g, epsilon))  # Ensure non-negative determinant

    # Compute Christoffel symbols (simplified for diagonal metric)
    d_gtt_dx = np.gradient(g_tt, dx, axis=0)
    d_gtt_dy = np.gradient(g_tt, dy, axis=1)
    d_gtt_dz = np.gradient(g_tt, dz, axis=2)

    # Simplified Ricci scalar components
    Gamma_x_tt = 0.5 * g_xx * d_gtt_dx
    Gamma_y_tt = 0.5 * g_yy * d_gtt_dy
    Gamma_z_tt = 0.5 * g_zz * d_gtt_dz
    Ricci_scalar = - (np.gradient(sqrt_g * Gamma_x_tt, dx, axis=0) +
                      np.gradient(sqrt_g * Gamma_y_tt, dy, axis=1) +
                      np.gradient(sqrt_g * Gamma_z_tt, dz, axis=2)) / sqrt_g

    return Ricci_scalar


# Function to update the metric tensor based on the clock state
def update_metric_tensor(clock_state, g_tt, g_xx, g_yy, g_zz, S, t):
    # Extract the clock state probabilities
    rho = DensityMatrix(clock_state)
    probabilities = rho.probabilities_dict()

    # Get probabilities for '0' and '1' states, default to 0 if not present
    P0 = probabilities.get('0', 0)
    P1 = probabilities.get('1', 0)

    # Example metric update rules based on clock state
    g_tt += P0 * 0.1 * S * np.sin(t)  # Update when clock is in state |0>
    g_tt -= P1 * 0.1 * S * np.cos(t)  # Update when clock is in state |1>
    g_xx -= P0 * 0.1 * S * np.sin(t)  # Negative update to g_xx
    g_xx += P1 * 0.1 * S * np.cos(t)  # Positive update to g_xx
    g_yy -= P0 * 0.1 * S * np.sin(t)  # Similar for g_yy and g_zz
    g_yy += P1 * 0.1 * S * np.cos(t)
    g_zz -= P0 * 0.1 * S * np.sin(t)
    g_zz += P1 * 0.1 * S * np.cos(t)

    # Allow metric components to become negative, potentially leading to negative curvature
    return g_tt, g_xx, g_yy, g_zz


# Simulation setup
S = initial_entanglement(X, Y, Z)  # Initial entanglement
g_tt, g_xx, g_yy, g_zz = quantum_metric_from_entanglement(S)  # Initial metric tensor

# Initialize the simulator
simulator = StatevectorSimulator()

# Run the simulation
ricci_scalars = []

for t in tqdm(range(timesteps), desc="PaW Simulation Progress"):
    # Evolve the clock state
    clock_state = clock_state.evolve(Operator(time_evolution_operator))

    # Update the entanglement distribution
    S = evolve_entanglement(S, t * dt)

    # Update the metric tensor based on the clock state
    g_tt, g_xx, g_yy, g_zz = update_metric_tensor(clock_state, g_tt, g_xx, g_yy, g_zz, S, t * dt)

    # Calculate Ricci scalar based on the updated metric tensor
    Ricci_scalar = calculate_ricci_scalar_3d(g_tt, g_xx, g_yy, g_zz, dx, dy, dz)
    ricci_scalars.append(Ricci_scalar)

# Amplify and plot results
scale_factor = 1.5
ricci_scalars = [rs * (1 + t / timesteps * scale_factor) for t, rs in enumerate(ricci_scalars)]

plt.figure(figsize=(8, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(ricci_scalars[::100])))

for i, (Ricci_scalar, color) in enumerate(zip(ricci_scalars[::100], colors)):
    slice_idx_x = X.shape[0] // 2
    plt.plot(y, Ricci_scalar[slice_idx_x, :, slice_idx_x], color=color, label=f"Time: {i * dt * 100:.2f}")

plt.legend()
plt.xlabel('Position')
plt.ylabel('Ricci Scalar')
plt.title('Ricci Scalar Evolution in the PaW Framework')
plt.show()
