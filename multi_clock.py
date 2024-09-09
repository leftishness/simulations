import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# =================================
# Function to create entangled states for multiple clocks
# =================================

def create_multi_clock_state(monogamy_degrees, num_clocks):
    """
    Create an initial entangled quantum state for a system of multiple clocks.

    Parameters:
    - monogamy_degrees (list of floats): List of monogamy degrees for each clock pair.
    - num_clocks (int): Number of clocks in the system.

    Returns:
    - state (ndarray): A complex-valued quantum state representing the entangled clocks.
    """
    state_size = 2 ** num_clocks  # For n clocks, the state vector has size 2^n
    state = np.zeros(state_size, dtype=complex)

    # Create an initial entangled state for two qubits
    alpha_1 = np.sqrt(monogamy_degrees[0])
    beta_1 = np.sqrt(1 - monogamy_degrees[0])
    state[0] = alpha_1
    state[-1] = beta_1

    # Entangle additional clocks
    for i in range(1, num_clocks):
        alpha_i = np.sqrt(monogamy_degrees[i % len(monogamy_degrees)])
        beta_i = np.sqrt(1 - monogamy_degrees[i % len(monogamy_degrees)])
        state[i] = alpha_i
        state[-(i + 1)] = beta_i

    return state


# ===========================================
# Page-Wootters time evolution for multiple clocks
# ===========================================

def multi_clock_time_evolution(t, state, monogamy_degrees, interaction_strengths):
    """
    Implement the Page-Wootters mechanism for time evolution in a multi-clock system.

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

    # Create the unitary evolution operator
    U = np.eye(len(state), dtype=complex)

    # Apply the Page-Wootters evolution
    for i in range(len(state)):
        interaction_factor = interaction_strengths[i % num_clocks] * (1 - monogamy_degrees[i % num_clocks])
        U[i, i] = np.exp(-1j * theta * interaction_factor)

    # Special interaction between Clock 2 and Clock 3 for entanglement coupling
    U[2, 3] = 0.5 * np.exp(-1j * theta * interaction_strengths[2])
    U[3, 2] = 0.5 * np.exp(1j * theta * interaction_strengths[2])

    # Apply the unitary operator to evolve the state
    evolved_state = U @ state

    return evolved_state


# ============================================
# Gravitational Time Dilation (Schwarzschild metric)
# ============================================

def gravitational_time_dilation(t, G, M, r, c):
    """
    Calculate the time dilation factor for a clock experiencing classical gravitational time dilation.

    Parameters:
    - t (int): Current time step.
    - G (float): Gravitational constant.
    - M (float): Mass of the gravitating body.
    - r (float): Radial distance from the center of the gravitating body.
    - c (float): Speed of light.

    Returns:
    - dilated_time (float): Time dilation factor based on the Schwarzschild metric.
    """
    time_dilation_factor = np.sqrt(1 - (2 * G * M) / (r * c ** 2))
    dilated_time = t * time_dilation_factor
    return dilated_time


# ====================================
# Simulation for multi-clock quantum time dilation
# ====================================

def simulate_multi_clock_time_dilation(monogamy_degrees, num_steps, num_clocks, interaction_strengths, G, M, r, c):
    """
    Simulate quantum time dilation in a system of multiple entangled clocks, with comparison to classical gravitational time dilation.

    Parameters:
    - monogamy_degrees (list of floats): List of monogamy degrees for each clock pair.
    - num_steps (int): Number of time steps for the simulation.
    - num_clocks (int): Number of clocks in the system.
    - interaction_strengths (list of floats): Interaction strength for each clock.
    - G (float): Gravitational constant.
    - M (float): Mass of the gravitating body.
    - r (float): Radial distance for gravitational time dilation.
    - c (float): Speed of light.

    Returns:
    - cumulative_phase_vals (list of lists): Cumulative phase values for each clock over time.
    - gravitational_times (list): Time dilation values for the reference clock experiencing gravitational time dilation.
    """
    # Create the initial entangled state
    state = create_multi_clock_state(monogamy_degrees, num_clocks)

    # Initialize cumulative phases and storage for each clock
    cumulative_phases = np.zeros(num_clocks)
    cumulative_phase_vals = [[] for _ in range(num_clocks)]
    gravitational_times = []

    # Time evolution loop
    for step in tqdm(range(num_steps), desc="Simulating Time Dilation"):
        # Evolve the quantum state
        evolved_state = multi_clock_time_evolution(step, state, monogamy_degrees, interaction_strengths)

        # Calculate and accumulate the phase for each clock
        for clock in range(num_clocks):
            phase_step = np.angle(evolved_state[clock])
            # Ensure phase accumulation is always positive
            phase_step = np.abs(phase_step)
            cumulative_phases[clock] += phase_step
            cumulative_phase_vals[clock].append(cumulative_phases[clock])

        # Calculate the gravitational time dilation for the reference clock
        grav_time = gravitational_time_dilation(step, G, M, r, c)
        gravitational_times.append(grav_time)

    return cumulative_phase_vals, gravitational_times


# =================================
# Updated Plotting the cumulative phase results with gravitational time dilation
# =================================

def plot_multi_clock_phase_with_gravity(cumulative_phase_vals, num_clocks, monogamy_degrees, interaction_strengths,
                                        gravitational_times):
    """
    Plot the cumulative phase over time for each clock in the multi-clock system along with a reference clock experiencing gravitational time dilation.
    The legend includes information about each clock's degree of entanglement monogamy and interaction strength.

    Parameters:
    - cumulative_phase_vals (list of lists): Cumulative phase values for each clock over time.
    - num_clocks (int): Number of clocks in the system.
    - monogamy_degrees (list of floats): List of monogamy degrees for each clock pair.
    - interaction_strengths (list of floats): List of interaction strengths for each clock.
    - gravitational_times (list): Time dilation values for the reference clock experiencing gravitational time dilation.
    """
    plt.figure(figsize=(10, 6))

    for clock in range(num_clocks):
        plt.plot(cumulative_phase_vals[clock],
                 label=f'Clock {clock + 1} (Monogamy: {monogamy_degrees[clock]:.1f}, Interaction: {interaction_strengths[clock]:.2f})')

    # Plot the gravitational time dilation as a reference
    plt.plot(gravitational_times, label='Reference Clock (Gravitational Time Dilation)', linestyle='--', color='black')

    plt.title('Cumulative Phase of Multi-Clock System with Gravitational Time Dilation')
    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Phase / Gravitational Time')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ================================
# Main function to run the simulation
# ================================

def main():
    """
    Main function to run the quantum time dilation simulation for multiple clocks, with comparison to gravitational time dilation.
    """
    # Define the distinct degrees of entanglement monogamy for each clock pair
    monogamy_degrees = [0.2, 0.4, 0.6, 0.8]

    # Define the number of time steps, number of clocks, and distinct interaction strengths
    num_steps = 100
    num_clocks = 4
    interaction_strengths = [0.1, 0.2, 0.15, 0.3]  # Each clock has a different interaction strength

    # Gravitational constants for Schwarzschild time dilation
    G = 6.67430e-11  # m^3 kg^(-1) s^(-2), gravitational constant
    M = 5.972e24  # kg, mass of Earth
    r = 6.371e6  # m, radius of Earth
    c = 3e8  # m/s, speed of light

    # Run the simulation
    sim_results, gravitational_times = simulate_multi_clock_time_dilation(monogamy_degrees, num_steps, num_clocks,
                                                                          interaction_strengths, G, M, r, c)

    # Plot the cumulative phase results, with the gravitational reference clock
    plot_multi_clock_phase_with_gravity(sim_results, num_clocks, monogamy_degrees, interaction_strengths,
                                        gravitational_times)


# ================================
# Run the simulation
# ================================

if __name__ == "__main__":
    main()
