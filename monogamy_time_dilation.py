import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import StatevectorSimulator
from qiskit.quantum_info import partial_trace, entropy, Statevector
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar

# Function to create the entangled quantum state with a specified degree of monogamy
def create_entangled_state(monogamy_degree):
    """
    Creates an entangled quantum state for three qubits (Clock A, Clock B, Clock C)
    with the specified degree of entanglement monogamy.

    Parameters:
        monogamy_degree (float): The degree of entanglement monogamy.
                                 0.01 represents low monogamy,
                                 0.99 represents high monogamy.
    Returns:
        QuantumCircuit: A quantum circuit representing the entangled state.
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

# Function to generate the time evolution operator for the clocks
def time_evolution_operator(t, monogamy_degree, interaction_strength=0.1, damping_factor=0.01):
    """
    Generates a time evolution operator for the quantum system with the given interaction strength,
    damping factor, and degree of entanglement monogamy.

    Parameters:
        t (float): The current time step.
        monogamy_degree (float): Degree of entanglement monogamy.
        interaction_strength (float): Strength of the interactions between the clocks.
        damping_factor (float): Damping factor applied to smooth the phase evolution over time.
    Returns:
        np.ndarray: A time evolution matrix for the system of three clocks.
    """
    theta = (interaction_strength * (1 - monogamy_degree)) * t * np.exp(-damping_factor * t)

    # Generate the time evolution matrix for the system
    interaction_matrix = np.kron(
        np.kron(
            np.array([[np.exp(-1j * theta), 0], [0, np.exp(1j * theta)]]),  # Clock A
            np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])  # Clock B
        ),
        np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])  # Clock C
    )
    return interaction_matrix

# Function to simulate the time evolution of the quantum clocks
def simulate_entanglement_monogamy(monogamy_degree, num_steps, interaction_strength=0.1, damping_factor=0.01):
    """
    Simulates the time evolution of a system of quantum clocks under a specific degree of
    entanglement monogamy, and returns the evolution of phase differences over time.

    Parameters:
        monogamy_degree (float): Degree of entanglement monogamy (0.01 is low, 0.99 is high).
        num_steps (int): Number of time steps for the simulation.
        interaction_strength (float): Strength of the interactions between clocks.
        damping_factor (float): Factor used to dampen chaotic fluctuations in phase evolution.

    Returns:
        list: Evolution of the phase differences between the clocks over time.
    """
    # Create the initial quantum state with the given monogamy constraints
    qc = create_entangled_state(monogamy_degree)
    simulator = StatevectorSimulator()

    phase_diffs = []  # List to store phase differences over time
    cumulative_phase = 0  # Cumulative phase variable to track total phase evolution
    initial_state = simulator.run(qc).result().get_statevector()  # Run the simulation for the initial state

    # Simulate time evolution for each time step
    for step in range(num_steps):
        # Evolve the quantum state according to the time evolution operator
        evolved_state_array = time_evolution_operator(step * 0.5, monogamy_degree, interaction_strength, damping_factor) @ initial_state.data
        evolved_state = Statevector(evolved_state_array)

        # Calculate phase difference for Clock A (trace out Clocks B and C)
        reduced_clock_A = partial_trace(evolved_state, [1, 2])  # Partial trace over Clocks B and C
        phase_diff = np.angle(reduced_clock_A.data[1, 0]) - np.angle(reduced_clock_A.data[0, 0])  # Phase difference calculation
        cumulative_phase += phase_diff
        phase_diffs.append(phase_diff)

        # Update the initial state for the next iteration
        initial_state = evolved_state

    return phase_diffs

# Function to plot the results
def plot_results(results, monogamy_degrees, phase_change_rates_high_res, monogamy_degrees_high_res):
    """
    Plots the results of the quantum time dilation simulation, including cumulative entanglement entropy,
    cumulative phase, and average phase change rate vs monogamy degree.

    Parameters:
        results (list): Simulation results including cumulative entropy and phase.
        monogamy_degrees (list): List of monogamy degrees used in the simulation.
        phase_change_rates_high_res (list): Average phase change rates for high-resolution monogamy degrees.
        monogamy_degrees_high_res (list): High-resolution monogamy degrees.
    """
    plt.figure(figsize=(18, 6))  # Set figure size for horizontal layout

    # Plot 1: Cumulative Entanglement Entropy
    plt.subplot(1, 3, 1)
    for i, degree in enumerate(monogamy_degrees):
        plt.plot(results[i][1], label=f'Monogamy Degree: {degree}')
    plt.title('Cumulative Entanglement Entropy')
    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Entropy')
    plt.legend()

    # Plot 2: Cumulative Phase
    plt.subplot(1, 3, 2)
    for i, degree in enumerate(monogamy_degrees):
        plt.plot(results[i][0], label=f'Monogamy Degree: {degree}')
    plt.title('Cumulative Phase: Time Evolution')
    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Phase')
    plt.legend()

    # Plot 3: Average Phase Change Rate vs Monogamy Degree
    plt.subplot(1, 3, 3)
    plt.plot(monogamy_degrees_high_res, phase_change_rates_high_res, 'o', label='Data', color='g')
    plt.title('Average Phase Change Rate vs Monogamy Degree')
    plt.xlabel('Monogamy Degree')
    plt.ylabel('Average Phase Change Rate')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main function to execute the simulation and plotting
def main():
    # Low-Resolution Monogamy Degrees (for Cumulative Plots)
    monogamy_degrees = [0.01, 0.5, 0.99]  # Low, moderate, and high monogamy
    num_steps = 100  # Number of time steps
    interaction_strength = 0.1
    damping_factor = 0.01

    results = []

    # Simulate for Low-Resolution Monogamy Degrees (Cumulative Entropy and Phase)
    for degree in tqdm(monogamy_degrees, desc="Low-Res Simulation Progress"):
        phase_diffs = simulate_entanglement_monogamy(degree, num_steps, interaction_strength, damping_factor)
        entropies = np.cumsum(np.abs(phase_diffs))  # Simulated entropies
        cumulative_phase = np.cumsum(phase_diffs)
        results.append((cumulative_phase, entropies))

    # High-Resolution Monogamy Degrees (for Phase Change Rate)
    monogamy_degrees_high_res = np.linspace(0.01, 0.99, 5000)
    phase_change_rates_high_res = []

    # Simulate for High-Resolution Monogamy Degrees
    for degree in tqdm(monogamy_degrees_high_res, desc="High-Res Simulation Progress"):
        phase_diffs = simulate_entanglement_monogamy(degree, num_steps, interaction_strength, damping_factor)

        # Phase Change Rate (use average over time steps)
        phase_change_rate = np.mean(np.abs(np.diff(phase_diffs)))
        phase_change_rates_high_res.append(phase_change_rate)

    # Plot all results
    plot_results(results, monogamy_degrees, phase_change_rates_high_res, monogamy_degrees_high_res)

if __name__ == "__main__":
    main()
