import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import StatevectorSimulator
from qiskit.quantum_info import state_fidelity, partial_trace, entropy
from tqdm import tqdm

# Constants
G = 6.67430e-11  # Gravitational constant
c = 299792458  # Speed of light
hbar = 1.0545718e-34  # Reduced Planck constant

# System parameters
M = 1.989e30  # Mass of the Sun
r_s = 2 * G * M / c**2  # Schwarzschild radius of the Sun
r_range = np.logspace(0, 3, 100) * r_s  # Distance range, scaled to Schwarzschild radii
g_max = 5.0  # Maximum entanglement strength

t_range = np.linspace(0, 50, 200)  # Time range

# Function to calculate classical time dilation
def classical_time_dilation(r):
    return np.sqrt(1 - 2 * G * M / (r * c ** 2))

# Function to calculate quantum time dilation from fidelity
def quantum_time_dilation(fidelity):
    return 1 - fidelity  # Time dilation increases as fidelity decreases

# Function to calculate von Neumann entropy for time dilation
def quantum_entropy(state):
    reduced_state = partial_trace(state, [0])  # Trace out one qubit
    return entropy(reduced_state)

# Link entanglement strength to physical distance from gravitational source
def entanglement_strength_from_mass_distance(M, r):
    return np.clip(g_max * (G * M / (r * c**2)), 0, g_max)  # Scaled entanglement strength

# Quantum-corrected spacetime curvature from entanglement
def quantum_corrected_metric(r, g):
    classical_metric = 1 - 2 * G * M / (r * c ** 2)
    quantum_correction = np.exp(-g)  # Model: curvature correction decreases with stronger entanglement
    return classical_metric * (1 + quantum_correction)

# Create quantum circuit for multipartite system (3 qubits)
def create_clock_circuit(g, t, num_clocks=3, initial_entanglement='GHZ'):
    qr = QuantumRegister(num_clocks)
    qc = QuantumCircuit(qr)

    # GHZ-like state initialization
    if initial_entanglement == 'GHZ':
        qc.h(0)
        for i in range(num_clocks - 1):
            qc.cx(i, i + 1)
    elif initial_entanglement == 'W':
        qc.x(num_clocks - 1)
        for i in range(num_clocks - 1):
            qc.cswap(num_clocks - 1, i, i + 1)  # W-state preparation
    else:
        raise ValueError("Unsupported initial state")

    # Evolution with dynamic interaction strength
    steps = 50
    dt = t / steps
    for _ in range(steps):
        for i in range(num_clocks):
            qc.rx(2 * g * dt, i)

    return qc

# Simulation loop with tqdm progress bar
def run_simulation(M, r_values):
    simulator = StatevectorSimulator()

    # Arrays to store results
    fidelity_results = np.zeros((len(r_values), len(t_range)))
    entropy_results = np.zeros((len(r_values), len(t_range)))

    # GHZ initial state for fidelity calculation
    initial_qc = create_clock_circuit(0, 0, num_clocks=3, initial_entanglement='GHZ')
    initial_result = simulator.run(transpile(initial_qc, simulator)).result()
    initial_state = initial_result.get_statevector()

    # GHZ initial state
    for i, r in tqdm(enumerate(r_values), total=len(r_values), desc="Running simulation"):
        g = entanglement_strength_from_mass_distance(M, r)
        for j, t in enumerate(t_range):
            # Create and run quantum circuit
            qc = create_clock_circuit(g, t, num_clocks=3, initial_entanglement='GHZ')
            transpiled_qc = transpile(qc, simulator)
            result = simulator.run(transpiled_qc).result()

            # Retrieve statevector and calculate fidelity and entropy
            final_state = result.get_statevector(transpiled_qc)
            fidelity = state_fidelity(initial_state, final_state)  # Compare state with initial state
            entropy_val = quantum_entropy(final_state)

            # Store results
            fidelity_results[i, j] = quantum_time_dilation(fidelity)
            entropy_results[i, j] = entropy_val

    return fidelity_results, entropy_results

# Classical time dilation comparison
def plot_classical_vs_quantum_dilation(r_values, g_values):
    classical_dilation = classical_time_dilation(r_values)
    quantum_dilation = [quantum_corrected_metric(r, g) for r, g in zip(r_values, g_values)]

    plt.figure(figsize=(8, 6))
    plt.plot(r_values / r_s, classical_dilation, label='Classical Time Dilation', linestyle='--')
    plt.plot(r_values / r_s, quantum_dilation, label='Quantum-Corrected Metric')
    plt.xlabel('r / r_s (Schwarzschild radii)')
    plt.ylabel('Time Dilation Factor / Metric Factor')
    plt.xscale('log')
    plt.title('Classical vs Quantum-Corrected Spacetime Dilation')
    plt.legend()
    plt.show()

# Main plotting function
def plot_results(r_values, time_steps, fidelity_results, entropy_results):
    # Fidelity plot
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(fidelity_results, aspect='auto', extent=[time_steps[0], time_steps[-1], r_values[-1], r_values[0]],
               cmap='viridis', vmin=0.0, vmax=1.0)
    plt.colorbar(label='Quantum Time Dilation (Fidelity)')
    plt.ylabel('Distance from Gravitational Source (r)')
    plt.xlabel('Time')
    plt.title('Quantum Time Dilation (Fidelity)')

    # Entropy plot
    plt.subplot(1, 2, 2)
    plt.imshow(entropy_results, aspect='auto', extent=[time_steps[0], time_steps[-1], r_values[-1], r_values[0]],
               cmap='plasma', vmin=-2, vmax=2)
    plt.colorbar(label='Quantum Time Dilation (Entropy)')
    plt.ylabel('Distance from Gravitational Source (r)')
    plt.xlabel('Time')
    plt.title('Quantum Time Dilation (Entropy)')

    plt.tight_layout()
    plt.show()

# Run the simulation with mass of the Sun and distance range
fidelity_results, entropy_results = run_simulation(M, r_range)

# Plot the results
plot_results(r_range, t_range, fidelity_results, entropy_results)

# Calculate entanglement strength based on distance and plot classical vs quantum dilation
g_values = [entanglement_strength_from_mass_distance(M, r) for r in r_range]
plot_classical_vs_quantum_dilation(r_range, g_values)
