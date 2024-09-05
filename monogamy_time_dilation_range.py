import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# =================================
# Function to create distinct entangled states for multiple clocks
# =================================

def create_multi_clock_state(monogamy_degrees, num_clocks):
    """
    Create an initial entangled quantum state for a system of multiple clocks,
    ensuring Clock 3 is properly entangled.

    Parameters:
    - monogamy_degrees (list of floats): List of monogamy degrees for each clock pair.
    - num_clocks (int): Number of clocks in the system.

    Returns:
    - state (ndarray): A complex-valued quantum state representing the entangled clocks.
    """
    state_size = 2 ** num_clocks  # For n clocks, the state vector has size 2^n
    state = np.zeros(state_size, dtype=complex)

    # Define pairwise entanglement
    alpha = np.sqrt(monogamy_degrees[0])
    beta = np.sqrt(1 - monogamy_degrees[0])
    state[0] = alpha
    state[-1] = beta

    # Ensure Clock 3 is entangled with Clock 2, and Clock 4 is entangled with Clock 1
    alpha_3 = np.sqrt(monogamy_degrees[2])
    beta_3 = np.sqrt(1 - monogamy_degrees[2])

    # Entangle Clock 3 with another clock
    state[2] = alpha_3
    state[5] = beta_3  # Adjusting the index to properly create entanglement

    return state

# ===========================================
# Page-Wootters time evolution (relational time) with distinct interaction strengths
# ===========================================

def multi_clock_time_evolution(t, state, monogamy_degrees, interaction_strengths):
    """
    Implement the Page-Wootters mechanism for time evolution in a multi-clock system.

    Ensure that Clock 3 has a non-zero interaction strength and is affected by the unitary evolution.

    Parameters:
    - t (int): Current time step.
    - state (ndarray): The quantum state vector at the current time.
    - monogamy_degrees (list of floats): List of monogamy degrees for each clock pair.
    - interaction_strengths (list of floats): Interaction strength for each clock.

    Returns:
    - evolved_state (ndarray): Evolved quantum state after applying the unitary operator at time t.
    """
    num_clocks = int(np.log2(len(state)))  # Number of clocks in the system
    theta = 2 * np.pi * t / 100  # Relational time scaling factor

    # Create the unitary evolution operator (generalized for multi-clock system)
    U = np.eye(len(state), dtype=complex)

    # Modify diagonal elements to simulate time evolution (with distinct interaction strengths)
    for i in range(len(state)):
        interaction_factor = interaction_strengths[i % len(interaction_strengths)] * (
                1 - monogamy_degrees[i % len(monogamy_degrees)])
        U[i, i] = np.exp(-1j * theta * interaction_factor)

    # Ensure interaction between Clock 2 and Clock 3 is properly set
    U[2, 3] = 0.5 * np.exp(-1j * theta * interaction_strengths[2])
    U[3, 2] = 0.5 * np.exp(1j * theta * interaction_strengths[2])

    # Apply the unitary operator to evolve the state
    return U @ state

# ====================================
# Simulation for multi-clock quantum time dilation with distinct parameters
# ====================================

def simulate_multi_clock_time_dilation(monogamy_degrees, num_steps, num_clocks, interaction_strengths):
    """
    Simulate the quantum time dilation effect in a system of multiple entangled clocks.
    Each clock has distinct entanglement monogamy and interaction strengths.
    """
    # Create the initial entangled state with distinct monogamy degrees
    initial_state = create_multi_clock_state(monogamy_degrees, num_clocks)

    # Initialize storage for cumulative phases of each clock
    cumulative_phases = np.zeros(num_clocks)
    cumulative_phase_vals = [[] for _ in range(num_clocks)]

    # Time evolution loop
    for step in range(num_steps):
        # Evolve the quantum state based on the Page-Wootters mechanism
        evolved_state = multi_clock_time_evolution(step, initial_state, monogamy_degrees, interaction_strengths)

        # Calculate the phase for each clock and accumulate the phase over time
        for clock in range(num_clocks):
            phase_step = np.angle(evolved_state[clock])
            cumulative_phases[clock] += phase_step
            cumulative_phase_vals[clock].append(cumulative_phases[clock])

    return cumulative_phase_vals

# =================================
# Plotting the cumulative phase
# =================================

def plot_multi_clock_phase(results, num_clocks):
    """
    Plot the cumulative phase over time for each clock in the multi-clock system.

    Parameters:
    - results (list of lists): Cumulative phase values for each clock.
    - num_clocks (int): Number of clocks in the system.
    """
    plt.figure(figsize=(10, 6))

    # Plot the cumulative phase for each clock using default line styles and colors
    for clock in range(num_clocks):
        plt.plot(results[clock], label=f'Clock {clock + 1}')

    plt.title('Cumulative Phase of Multi-Clock System (Page-Wootters Evolution)')
    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Phase')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ================================
# Main function to run the multi-clock simulation
# ================================

def main():
    """
    Main function to run the quantum time dilation simulation for multiple clocks.
    """
    # Define the distinct degrees of entanglement monogamy for each clock pair
    monogamy_degrees = [0.2, 0.4, 0.6, 0.8]  # Adjust this to vary monogamy constraints

    # Define the number of time steps, clocks, and distinct interaction strengths
    num_steps = 100
    num_clocks = 4  # For example, a system of 4 quantum clocks
    interaction_strengths = [0.1, 0.2, 0.15, 0.3]  # Different interaction strengths for each clock

    # Run the simulation
    sim_results = simulate_multi_clock_time_dilation(monogamy_degrees, num_steps, num_clocks, interaction_strengths)

    # Plot the cumulative phase results
    plot_multi_clock_phase(sim_results, num_clocks)


# ================================
# Run the simulation
# ================================

if __name__ == "__main__":
    main()
