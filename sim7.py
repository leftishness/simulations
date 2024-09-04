import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator, DensityMatrix, partial_trace
from qiskit_aer import AerSimulator
from scipy.linalg import expm

# Define Hamiltonians for single qubits
H_A = np.array([[0.5, 0.0], [0.0, -0.5]])  # Pauli-Z for Clock A
H_B = np.array([[0.5, 0.0], [0.0, -0.5]])  # Pauli-Z for Clock B (Reference)
H_int = np.kron(np.array([[0.0, 0.5], [0.5, 0.0]]), np.eye(2))  # Pauli-X interaction expanded for 2 qubits

def evolve_clock(state, hamiltonian, dt, timesteps):
    evolved_states = []
    for _ in range(timesteps):
        operator = Operator(expm(-1j * hamiltonian * dt))
        state = state.evolve(operator)
        evolved_states.append(state)
    return evolved_states

def normalize_phase(phases):
    """Normalize the phase evolution to the range [0, 1]."""
    min_phase = np.min(phases)
    max_phase = np.max(phases)
    return (phases - min_phase) / (max_phase - min_phase)

def calculate_phase_evolution(states, qubit=0):
    Z_op = np.array([[1, 0], [0, -1]])  # Pauli-Z operator
    expectation_values = []
    for state in states:
        dm = DensityMatrix(state)
        if len(dm.dims()) > 1:  # Check if there are multiple qubits
            qubit_dm = partial_trace(dm, [1]).data  # Convert to NumPy array
        else:
            qubit_dm = dm.data
        expectation_values.append(np.real(np.trace(Z_op @ qubit_dm)))

    # Normalize the phase evolution
    return normalize_phase(expectation_values)

def calculate_paw_time(state, hamiltonian, dt, timesteps):
    evolved_states = evolve_clock(state, hamiltonian, dt, timesteps)
    reduced_state = partial_trace(DensityMatrix(evolved_states[-1]), [1]).data  # Convert to NumPy array
    paw_time = -np.angle(reduced_state[0, 0]) / np.pi
    return paw_time

def calculate_monogamy_constraint(entanglement_A_C):
    """Calculate the remaining entanglement capacity due to the monogamy constraint."""
    return np.sqrt(1 - entanglement_A_C**2)

def run_simulation(simulator, entanglement_strengths, timesteps, dt):
    phases_A = []
    phases_B = []
    paw_times = []
    monogamy_constraints = []

    for entanglement_strength in entanglement_strengths:
        H_A_expanded = np.kron(H_A, np.eye(2)) + entanglement_strength * H_int
        H_B_expanded = np.kron(H_B, np.eye(2)) + entanglement_strength * H_int
        monogamy_constraint = calculate_monogamy_constraint(entanglement_strength)
        monogamy_constraints.append(monogamy_constraint)

        initial_state_AC = Statevector.from_label('00')
        initial_state_BC = Statevector.from_label('00')

        states_A = evolve_clock(initial_state_AC, H_A_expanded, dt, timesteps)
        states_B = evolve_clock(initial_state_BC, H_B_expanded, dt, timesteps)

        phases_A.append(calculate_phase_evolution(states_A, qubit=0))
        phases_B.append(calculate_phase_evolution(states_B, qubit=0))
        paw_times.append(calculate_paw_time(initial_state_AC, H_A_expanded, dt, timesteps))

    return phases_A, phases_B, paw_times, monogamy_constraints

def plot_results(phases_A, phases_B, paw_times, monogamy_constraints, entanglement_strengths):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot Phase Evolution of Clocks
    for i, phase in enumerate(phases_A):
        axs[0].plot(phase, label=f'Clock A (Ent. {entanglement_strengths[i]:.2f})')
    axs[0].plot(phases_B[0], '--', label='Clock B (Reference)')  # Only plot one reference line
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Phase Evolution')
    axs[0].legend()
    axs[0].set_title('Phase Evolution of Quantum Clocks')

    # Plot PaW Time Evolution against Monogamy Constraint
    axs[1].plot(entanglement_strengths, paw_times, 'o-', label='PaW Time')
    axs[1].set_xlabel('Entanglement Strength')
    axs[1].set_ylabel('PaW Time')
    axs[1].legend()
    axs[1].set_title('PaW Time Evolution')

    # Plot Monogamy Constraint
    axs[2].plot(entanglement_strengths, monogamy_constraints, 'm.-')
    axs[2].set_xlabel('Entanglement with System C')
    axs[2].set_ylabel('Remaining Entanglement Capacity')
    axs[2].set_title('Entanglement Monogamy Constraint')

    plt.tight_layout()
    plt.show()


def main():
    simulator = AerSimulator()
    entanglement_strengths = [0.0, 0.25, 0.5, 0.75, 0.99]
    timesteps = 100
    dt = 0.1

    phases_A, phases_B, paw_times, monogamy_constraints = run_simulation(simulator, entanglement_strengths, timesteps, dt)
    plot_results(phases_A, phases_B, paw_times, monogamy_constraints, entanglement_strengths)

if __name__ == "__main__":
    main()
