import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector, state_fidelity, partial_trace
from qiskit_aer import StatevectorSimulator
from qiskit.quantum_info import entropy, concurrence
from qiskit.visualization import plot_state_qsphere

# Constants
G = 6.67430e-11  # Gravitational constant
c = 299792458  # Speed of light
hbar = 1.0545718e-34  # Reduced Planck constant

# System parameters
M = 1.989e30  # Mass of the Sun
omega_C = 1.0  # Clock frequency
omega_R = 0.5  # Universe frequency
g_range = np.linspace(0, 10, 50)  # Range of interaction strengths
t_range = np.linspace(0, 20, 100)  # Time range

# Function to calculate classical time dilation
def classical_time_dilation(r):
    return np.sqrt(1 - 2 * G * M / (r * c ** 2))

# Function to calculate quantum time dilation from fidelity
def quantum_time_dilation(fidelity):
    return 1 - fidelity  # Time dilation increases as fidelity to initial state decreases

# Function to calculate von Neumann entropy for time dilation
def quantum_entropy(state):
    reduced_state = partial_trace(state, [0])  # Trace out one qubit to get reduced state
    return entropy(reduced_state)

# Create quantum circuit for multipartite system (3 qubits)
def create_circuit(g, t):
    qr = QuantumRegister(3)
    qc = QuantumCircuit(qr)

    # Initial GHZ-like state
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)

    # Evolution
    steps = 20
    dt = t / steps
    for _ in range(steps):
        qc.rz(omega_C * dt, 0)
        qc.rz(omega_R * dt, 1)
        qc.rx(2 * g * dt, 0)
        qc.rx(2 * g * dt, 1)
        qc.rx(2 * g * dt, 2)
        qc.cx(0, 1)
        qc.cx(1, 2)

    return qc

# Initialize simulator
simulator = StatevectorSimulator()

# Arrays to store results
gamma_q_array_fidelity = np.zeros((len(g_range), len(t_range)))
gamma_q_array_entropy = np.zeros((len(g_range), len(t_range)))

# Initial GHZ-like state
initial_state = Statevector([1 / np.sqrt(2), 0, 0, 0, 0, 0, 0, 1 / np.sqrt(2)])

# Simulation loop
for i, g in enumerate(g_range):
    for j, t in enumerate(t_range):
        # Create and run circuit
        qc = create_circuit(g, t)
        result = simulator.run(qc).result()
        final_state = result.get_statevector()

        # Calculate fidelity and quantum time dilation from fidelity
        fidelity = state_fidelity(initial_state, final_state)
        gamma_q_array_fidelity[i, j] = quantum_time_dilation(fidelity)

        # Calculate von Neumann entropy and quantum time dilation
        gamma_q_array_entropy[i, j] = quantum_entropy(final_state)

# Calculate classical time dilation
r_range = np.logspace(0, 4, 1000) * (2 * G * M / c ** 2)  # Range from event horizon to 10000 Schwarzschild radii
gamma_g = classical_time_dilation(r_range)

# Plotting
plt.figure(figsize=(12, 16))

# Quantum time dilation from fidelity
plt.subplot(311)
plt.imshow(gamma_q_array_fidelity, aspect='auto', extent=[t_range[0], t_range[-1], g_range[-1], g_range[0]],
           cmap='viridis', vmin=0, vmax=1)
plt.colorbar(label='Quantum Time Dilation Factor (Fidelity)')
plt.ylabel('Interaction Strength g')
plt.title('Quantum Time Dilation (Fidelity)')

# Quantum time dilation from entropy
plt.subplot(312)
plt.imshow(gamma_q_array_entropy, aspect='auto', extent=[t_range[0], t_range[-1], g_range[-1], g_range[0]],
           cmap='plasma', vmin=0, vmax=1)
plt.colorbar(label='Quantum Time Dilation Factor (Entropy)')
plt.ylabel('Interaction Strength g')
plt.title('Quantum Time Dilation (von Neumann Entropy)')

# Classical time dilation
plt.subplot(313)
plt.semilogx(r_range / (2 * G * M / c ** 2), gamma_g)
plt.xlabel('r / r_s (Schwarzschild radii)')
plt.ylabel('Classical Time Dilation Factor')
plt.title('Classical Gravitational Time Dilation')
plt.ylim(0, 1)

plt.tight_layout()
plt.show()

# Print some key values
print(f"Max Quantum Time Dilation (Fidelity): {np.max(gamma_q_array_fidelity):.4f}")
print(f"Max Quantum Time Dilation (Entropy): {np.max(gamma_q_array_entropy):.4f}")
print(f"Classical Time Dilation at r=2r_s: {classical_time_dilation(2 * (2 * G * M / c ** 2)):.4f}")

# Visualize final state for maximum interaction strength and time
final_qc = create_circuit(g_range[-1], t_range[-1])
final_statevector = simulator.run(final_qc).result().get_statevector()
plot_state_qsphere(final_statevector)
plt.show()

# Plot time evolution of quantum time dilation for different interaction strengths
plt.figure(figsize=(10, 6))
for g in [0, 2, 5, 10]:
    idx = np.argmin(np.abs(g_range - g))
    plt.plot(t_range, gamma_q_array_fidelity[idx, :], label=f'g = {g}')
plt.xlabel('Time')
plt.ylabel('Quantum Time Dilation Factor (Fidelity)')
plt.title('Time Evolution of Quantum Time Dilation (Fidelity)')
plt.legend()
plt.show()

# Time evolution for entropy-based dilation
plt.figure(figsize=(10, 6))
for g in [0, 2, 5, 10]:
    idx = np.argmin(np.abs(g_range - g))
    plt.plot(t_range, gamma_q_array_entropy[idx, :], label=f'g = {g}')
plt.xlabel('Time')
plt.ylabel('Quantum Time Dilation Factor (Entropy)')
plt.title('Time Evolution of Quantum Time Dilation (Entropy)')
plt.legend()
plt.show()
