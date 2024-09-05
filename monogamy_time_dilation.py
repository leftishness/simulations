import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import StatevectorSimulator
from qiskit.quantum_info import partial_trace, Statevector, DensityMatrix
from tqdm import tqdm  # Progress bar


# ========================
# FUNCTIONAL IMPLEMENTATION
# ========================

def create_entangled_state(monogamy_degree):
    """
    Create an entangled state between three quantum clocks (A, B, and C)
    with a specified degree of monogamy between clocks A and C.
    """
    qr = QuantumRegister(3)  # 0: Clock A, 1: Clock B, 2: Clock C
    qc = QuantumCircuit(qr)

    # Entangle Clock A and Clock B
    qc.h(0)  # Put Clock A into superposition
    qc.cx(0, 1)  # Entangle Clock A with Clock B

    # Adjust the entanglement between Clock A and Clock C based on the monogamy degree
    qc.ry(2 * np.arcsin(np.sqrt(monogamy_degree)), 2)  # Rotate Clock C based on monogamy degree
    qc.cx(0, 2)  # Entangle Clock A and Clock C (with monogamy constraint)

    return qc


def time_evolution_operator(t, monogamy_degree, interaction_strength=0.1, damping_factor=0.01):
    """
    Generate a time evolution operator for the quantum clocks using a simple interaction Hamiltonian.
    """
    theta = (interaction_strength * (1 - monogamy_degree)) * t * np.exp(-damping_factor * t)
    hamiltonian = np.array([[0, 1], [1, 0]])  # Pauli-X interaction Hamiltonian
    U = np.cos(theta) * np.eye(2) - 1j * np.sin(theta) * hamiltonian
    U_total = np.kron(np.kron(U, U), U)  # Apply to all qubits

    return U_total


def simulate_entanglement_monogamy(monogamy_degree, num_steps, interaction_strength=0.1, damping_factor=0.01):
    """
    Simulate the entanglement monogamy effect on time dilation in a system of quantum clocks.
    """
    qc = create_entangled_state(monogamy_degree)
    simulator = StatevectorSimulator()
    initial_state = simulator.run(transpile(qc, simulator)).result().get_statevector()

    cumulative_phases = []
    phase_diffs = []
    cumulative_phase = 0

    for step in range(num_steps):
        evolved_state_array = time_evolution_operator(step * 0.5, monogamy_degree, interaction_strength,
                                                      damping_factor) @ initial_state.data
        evolved_state = Statevector(evolved_state_array)

        reduced_state_A = partial_trace(evolved_state, [1, 2])
        reduced_state_A_data = DensityMatrix(reduced_state_A).data

        # Calculate phase difference for Clock A's state
        phase_diff_A = np.angle(evolved_state[0]) - np.angle(evolved_state[1])
        phase_diffs.append(phase_diff_A)  # Log instantaneous phase difference
        cumulative_phase += phase_diff_A
        cumulative_phases.append(cumulative_phase)
        initial_state = evolved_state

    return cumulative_phases, phase_diffs


def plot_cumulative_phase_evolution(results, monogamy_degrees):
    """
    Plot the cumulative phase for two extreme monogamy degrees to highlight time dilation.
    """
    plt.figure(figsize=(8, 6))

    for i, degree in enumerate(monogamy_degrees):
        plt.plot(results[i], label=f'Monogamy Degree: {degree}')
    plt.title('Cumulative Phase: Time Evolution (Extremes)')
    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Phase')

    # Dynamically adjust axis limits to better visualize small changes
    all_phases = np.concatenate(results)
    plt.ylim(min(all_phases) - 0.01, max(all_phases) + 0.01)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_instantaneous_phase_change(phase_diffs, monogamy_degrees):
    """
    Plot the instantaneous phase change at each time step for two monogamy extremes.
    """
    plt.figure(figsize=(8, 6))

    for i, diffs in enumerate(phase_diffs):
        plt.plot(diffs, label=f'Monogamy Degree: {monogamy_degrees[i]}')
    plt.title('Instantaneous Phase Change vs Time Step')
    plt.xlabel('Time Steps')
    plt.ylabel('Instantaneous Phase Change')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_cumulative_phase_vs_monogamy(high_res_monogamy_degrees, high_res_phase_changes):
    """
    Plot total cumulative phase change vs monogamy degree with a curve fit to show clear trend.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(high_res_monogamy_degrees, high_res_phase_changes, label='Total Cumulative Phase Change', color='g')

    # Fit a polynomial curve (degree 2) to show the trend
    def poly_fit(x, a, b, c):
        return a * x ** 2 + b * x + c

    popt, _ = curve_fit(poly_fit, high_res_monogamy_degrees, high_res_phase_changes)
    fit_curve = poly_fit(high_res_monogamy_degrees, *popt)

    # Plot the fitted curve
    plt.plot(high_res_monogamy_degrees, fit_curve, color='r', label='Fitted Curve', linewidth=2)

    plt.title('Total Cumulative Phase Change vs Monogamy Degree (with Fit)')
    plt.xlabel('Monogamy Degree')
    plt.ylabel('Total Cumulative Phase Change')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ========================
# MAIN FUNCTION
# ========================
def main():
    # Low-Resolution Monogamy Degrees (Focus on extremes)
    monogamy_degrees = [0.01, 0.99]  # Two extremes
    num_steps = 100
    interaction_strength = 0.1
    damping_factor = 0.01

    results = []
    phase_diffs = []

    # Simulate for the two extreme Monogamy Degrees (Cumulative Phase)
    for degree in tqdm(monogamy_degrees, desc="Low-Res Simulation Progress"):
        cumulative_phase, diffs = simulate_entanglement_monogamy(degree, num_steps, interaction_strength,
                                                                 damping_factor)
        results.append(cumulative_phase)
        phase_diffs.append(diffs)

    # High-Resolution Monogamy Degrees (for Cumulative Phase Change vs Monogamy Degree)
    high_res_monogamy_degrees = np.linspace(0.01, 0.99, 5000)  # 50 data points for high resolution
    high_res_phase_changes = []

    # Simulate for High-Resolution Monogamy Degrees
    for degree in tqdm(high_res_monogamy_degrees, desc="High-Res Phase Change Calculation"):
        cumulative_phase, _ = simulate_entanglement_monogamy(degree, num_steps, interaction_strength, damping_factor)
        total_phase_change = np.sum(cumulative_phase)
        high_res_phase_changes.append(total_phase_change)

    # Plot Cumulative Phase for extreme Monogamy Degrees
    plot_cumulative_phase_evolution(results, monogamy_degrees)

    # Plot Total Cumulative Phase Change vs Monogamy Degree (with curve fit)
    plot_cumulative_phase_vs_monogamy(high_res_monogamy_degrees, high_res_phase_changes)

    # Plot Instantaneous Phase Change for extreme Monogamy Degrees
    plot_instantaneous_phase_change(phase_diffs, monogamy_degrees)


if __name__ == "__main__":
    main()
