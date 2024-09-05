import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# =================================
# Function to create entangled state
# =================================

def create_entangled_state(monogamy_degree):
    """
    Create an initial strongly entangled quantum state between
    clock A and environment (B and C) based on the monogamy degree.

    The stronger the monogamy degree, the more constrained the
    entanglement between A and B, leading to a slower clock evolution.

    Parameters:
    - monogamy_degree (float): A value between 0 and 1 representing
      the degree of monogamy (0 = minimal, 1 = maximal entanglement).

    Returns:
    - state (ndarray): A complex-valued 8-dimensional quantum state.
    """
    alpha = np.sqrt(monogamy_degree)
    beta = np.sqrt(1 - monogamy_degree)
    return np.array([alpha, 0, 0, beta, beta, 0, 0, alpha], dtype=complex)


# ===========================================
# Page-Wootters time evolution (relational time)
# ===========================================

def page_wootters_time_evolution(t, state, monogamy_degree, interaction_strength=0.1):
    """
    Implement the Page-Wootters mechanism for time evolution based on relational time.

    The unitary evolution operator includes stronger interaction terms
    for higher entanglement, simulating quantum time dilation.

    Parameters:
    - t (int): Current time step.
    - state (ndarray): The quantum state vector at the current time.
    - monogamy_degree (float): Degree of entanglement monogamy between
      subsystems A, B, and C.
    - interaction_strength (float): Scaling factor for the strength of interactions
      in the time evolution.

    Returns:
    - evolved_state (ndarray): Evolved quantum state after applying
      the unitary operator at time step t.
    """
    theta = 2 * np.pi * t / 100  # Relational time scaling factor
    interaction_factor = interaction_strength * (1 - monogamy_degree)

    # Unitary time evolution operator (Pauli-X-based interaction)
    U = np.eye(8, dtype=complex)
    U[0, 0] = np.exp(-1j * theta * interaction_factor)
    U[7, 7] = np.exp(-1j * theta * interaction_factor)
    U[3, 3] = U[4, 4] = np.exp(1j * theta * interaction_factor)

    # Adding cross terms for stronger entanglement dynamics
    U[1, 6] = U[6, 1] = 0.5 * np.exp(-1j * theta * interaction_factor)
    U[2, 5] = U[5, 2] = 0.5 * np.exp(1j * theta * interaction_factor)

    # Apply the unitary operator to the current state
    return U @ state


# =================================
# Function to calculate phase
# =================================

def calculate_phase(state):
    """
    Calculate the cumulative phase of the quantum clock (Clock A).

    The phase of the clock's quantum state provides a measure of how
    fast time is evolving for that clock. This is the primary indicator
    of quantum time dilation.

    Parameters:
    - state (ndarray): The quantum state vector at the current time.

    Returns:
    - phase (float): The phase angle of the quantum state of clock A.
    """
    return np.angle(state[0])


# ====================================
# Simulation for quantum time dilation
# ====================================

def simulate_time_dilation(monogamy_degree, num_steps, interaction_strength=0.1):
    """
    Simulate the quantum time dilation effect for a system of entangled clocks.

    The evolution of the system is governed by the Page-Wootters relational
    time mechanism, and the cumulative phase is tracked to capture the time
    dilation effect based on entanglement monogamy.

    Parameters:
    - monogamy_degree (float): Degree of entanglement monogamy.
    - num_steps (int): Number of time steps for the simulation.
    - interaction_strength (float): Scaling factor for the strength of interactions.

    Returns:
    - cumulative_phase_vals (list): List of cumulative phase values over time.
    """
    # Create the initial entangled state
    initial_state = create_entangled_state(monogamy_degree)

    cumulative_phase_vals = []  # To store cumulative phase values
    cumulative_phase = 0  # Initialize the cumulative phase

    # Time evolution loop
    for step in range(num_steps):
        # Evolve the quantum state based on the Page-Wootters mechanism
        evolved_state = page_wootters_time_evolution(step, initial_state, monogamy_degree, interaction_strength)

        # Calculate the phase for the current time step
        phase_step = calculate_phase(evolved_state)

        # Accumulate the phase over time
        cumulative_phase += phase_step
        cumulative_phase_vals.append(cumulative_phase)

    return cumulative_phase_vals


# ================================
# Plotting the cumulative phase
# ================================

def plot_cumulative_phase(results, monogamy_degrees):
    """
    Plot the cumulative phase over time for different degrees of entanglement monogamy.

    The cumulative phase indicates how much the clock's state has evolved
    over time, with slower evolution (greater time dilation) at higher monogamy.

    Parameters:
    - results (list of lists): Cumulative phase values for different monogamy degrees.
    - monogamy_degrees (list): List of monogamy degree values used in the simulation.
    """
    plt.figure(figsize=(10, 6))

    # Plot the cumulative phase for each monogamy degree
    for i, degree in enumerate(monogamy_degrees):
        plt.plot(results[i], label=f'Monogamy Degree: {degree}')

    plt.title('Cumulative Phase of Clock A (Page-Wootters Evolution)')
    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Phase')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ================================
# Main function to run the simulation
# ================================

def main():
    """
    Main function to run the quantum time dilation simulation for different
    degrees of entanglement monogamy. The results are plotted to visualize
    the cumulative phase evolution (quantum time dilation).
    """
    # Define the degrees of entanglement monogamy to test
    monogamy_degrees = [0.01, 0.5, 0.99]

    # Define the number of time steps and interaction strength
    num_steps = 100
    interaction_strength = 0.2  # Stronger interactions for more pronounced entanglement effects

    results = []  # Store the results for each monogamy degree

    # Run the simulation for each monogamy degree
    for degree in tqdm(monogamy_degrees, desc="Simulating Time Dilation"):
        sim_results = simulate_time_dilation(degree, num_steps, interaction_strength)
        results.append(sim_results)

    # Plot the cumulative phase results
    plot_cumulative_phase(results, monogamy_degrees)


# ================================
# Run the simulation
# ================================

if __name__ == "__main__":
    main()
