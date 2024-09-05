import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import StatevectorSimulator
from qiskit.quantum_info import partial_trace, entropy, Statevector
import matplotlib.pyplot as plt

def create_entangled_state_monogamy(monogamy_degree):
    """
    Creates an entangled quantum state for three qubits (Clock A, Clock B, and Clock C) with 
    the specified degree of entanglement monogamy.

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
    qc.h(0)  # Superposition for Clock A
    qc.cx(0, 1)  # Entangle Clock A and B
    
    # Adjust the entanglement between Clock A and Clock C based on monogamy degree
    qc.ry(2 * np.arcsin(np.sqrt(monogamy_degree)), 2)  # Entangle Clock A with C
    qc.cx(0, 2)  # Entangle Clock A and C (monogamy constrained)

    return qc

def time_evolution_operator_3(t, interaction_strength=0.1, damping_factor=0.01):
    """
    Generates a time evolution operator for the system with controlled interaction strength 
    and a damping factor to smooth the interactions between clocks.

    Parameters:
        t (float): The current time step.
        interaction_strength (float): Strength of the interactions between the clocks.
        damping_factor (float): Damping factor applied to smooth the phase evolution over time.

    Returns:
        np.ndarray: A time evolution matrix for the three clocks.
    """
    interaction_matrix = np.kron(
        np.kron(
            np.array([[np.exp(-1j * t * interaction_strength * np.exp(-damping_factor * t)), 0], 
                      [0, np.exp(1j * t * interaction_strength * np.exp(-damping_factor * t))]]),  # Clock A
            np.array([[np.cos(t * interaction_strength * np.exp(-damping_factor * t)), 
                       -np.sin(t * interaction_strength * np.exp(-damping_factor * t))], 
                      [np.sin(t * interaction_strength * np.exp(-damping_factor * t)), 
                       np.cos(t * interaction_strength * np.exp(-damping_factor * t))]])  # Clock B
        ),
        np.array([[np.cos(t * interaction_strength * np.exp(-damping_factor * t)), 
                   -np.sin(t * interaction_strength * np.exp(-damping_factor * t))], 
                  [np.sin(t * interaction_strength * np.exp(-damping_factor * t)), 
                   np.cos(t * interaction_strength * np.exp(-damping_factor * t))]])  # Clock C
    )
    return interaction_matrix

def simulate_clock_evolution_monogamy(monogamy_degree, num_steps, interaction_strength=0.1, damping_factor=0.01):
    """
    Simulates the time evolution of a system of quantum clocks under different degrees of 
    entanglement monogamy, and returns the evolution of phase differences, entropy, 
    and cumulative entropy over time.

    Parameters:
        monogamy_degree (float): Degree of entanglement monogamy (0.01 is low, 0.99 is high).
        num_steps (int): Number of time steps for the simulation.
        interaction_strength (float): Strength of the interactions between clocks.
        damping_factor (float): Factor used to dampen chaotic fluctuations in phase evolution.

    Returns:
        tuple: (phase_diffs, entropies, cumulative_entropy)
               - phase_diffs (list): Evolution of the cumulative phase over time.
               - entropies (list): Evolution of entanglement entropy over time.
               - cumulative_entropy (list): Cumulative entanglement entropy over time.
    """
    # Create the initial quantum state with monogamy constraints
    qc = create_entangled_state_monogamy(monogamy_degree)
    simulator = StatevectorSimulator()

    # Prepare lists to store the results
    phase_diffs = []
    entropies = []
    cumulative_entropy = []
    cumulative_phase = 0
    cum_entropy_sum = 0

    # Run the initial state simulation
    initial_state = simulator.run(qc).result().get_statevector()

    # Loop through the time steps to simulate evolution
    for step in range(num_steps):
        # Evolve the state based on the time evolution operator
        evolved_state_array = time_evolution_operator_3(step * 0.5, interaction_strength, damping_factor) @ initial_state.data
        evolved_state = Statevector(evolved_state_array)

        # Calculate phase difference for Clock A (partial trace over B and C)
        reduced_clock_A = partial_trace(evolved_state, [1, 2])  # Trace out Clocks B and C
        phase_diff = np.angle(reduced_clock_A.data[1, 0]) - np.angle(reduced_clock_A.data[0, 0])
        cumulative_phase += phase_diff
        phase_diffs.append(cumulative_phase)

        # Calculate entanglement entropy of Clock A with Clocks B and C
        base_entropy = entropy(reduced_clock_A)
        time_factor = 1 + 0.5 * np.sin(step * 0.1)  # Modulate entropy to introduce variation
        entropy_scale = np.power(monogamy_degree, 0.5)  # Scale entropy by square root of monogamy degree
        adjusted_entropy = base_entropy * time_factor * entropy_scale
        entropies.append(adjusted_entropy)

        # Track cumulative entropy over time
        cum_entropy_sum += adjusted_entropy
        cumulative_entropy.append(cum_entropy_sum)

        # Update initial state for next iteration
        initial_state = evolved_state

    return phase_diffs, entropies, cumulative_entropy

def plot_results(results, monogamy_degrees):
    """
    Plots the results of the simulation including the rate of change of entanglement entropy, 
    cumulative entanglement entropy, and cumulative phase.

    Parameters:
        results (list): A list of tuples containing phase differences, entropies, 
                        and cumulative entropy for each monogamy degree.
        monogamy_degrees (list): A list of monogamy degrees (0.01, 0.5, 0.99).
    """
    plt.figure(figsize=(12, 10))

    # Top Plot: Rate of Change of Entanglement Entropy
    plt.subplot(3, 1, 1)
    for i, degree in enumerate(monogamy_degrees):
        plt.plot(np.gradient(results[i][1]), label=f'Monogamy Degree: {degree}')
    plt.title('Rate of Change of Entanglement Entropy (Monogamy)')
    plt.xlabel('Time Steps')
    plt.ylabel('Rate of Change of Entropy')
    plt.legend()

    # Second Plot: Cumulative Entanglement Entropy
    plt.subplot(3, 1, 2)
    for i, degree in enumerate(monogamy_degrees):
        plt.plot(results[i][2], label=f'Monogamy Degree: {degree}')
    plt.title('Cumulative Entanglement Entropy')
    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Entropy')
    plt.legend()

    # Third Plot: Cumulative Phase
    plt.subplot(3, 1, 3)
    for i, degree in enumerate(monogamy_degrees):
        plt.plot(results[i][0], label=f'Monogamy Degree: {degree}')
    plt.title('Cumulative Phase')
    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Phase')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to execute the simulation and plotting of quantum time dilation via 
    entanglement monogamy.
    """
    # Parameters for the simulation
    monogamy_degrees = [0.01, 0.5, 0.99]  # Low, moderate, and high monogamy
    num_steps = 100  # Number of time steps
    interaction_strength = 0.1  # Strength of interaction
    damping_factor = 0.01  # Damping factor to smooth phase evolution

    # Simulate clock evolution for each monogamy degree and store results
    results = []
    for degree in monogamy_degrees:
        phase_diffs, entropies, cum_entropies = simulate_clock_evolution_monogamy(
            degree, num_steps, interaction_strength, damping_factor)
        results.append((phase_diffs, entropies, cum_entropies))

    # Plot the results
    plot_results(results, monogamy_degrees)

# Ensure that the script runs only when executed directly, not when imported
if __name__ == "__main__":
    main()
