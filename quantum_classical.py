import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from scipy import stats
import seaborn as sns

# ===========================================
# Quantum Time Dilation Mechanisms
# ===========================================

def quantum_tick_rate(monogamy_degree, interaction_strength):
    """
    Calculate the quantum time dilation rate based on entanglement monogamy and interaction strength.
    
    Parameters:
    - monogamy_degree (float): Degree of entanglement monogamy (range [0, 1]).
    - interaction_strength (float): Interaction strength between qubits/clocks (range [0, 1]).

    Returns:
    - Quantum tick rate as a TensorFlow tensor.
    """
    return tf.exp(-monogamy_degree * interaction_strength)

def kinetic_tick_rate(velocity):
    """
    Calculate the relativistic time dilation due to velocity (Lorentz time dilation).
    
    Parameters:
    - velocity (float): Velocity as a fraction of the speed of light (range [0, 0.99]).

    Returns:
    - Kinetic time dilation factor based on special relativity.
    """
    return tf.sqrt(1 - tf.square(velocity))

def gravitational_tick_rate(gravitational_potential):
    """
    Calculate the classical time dilation due to gravitational potential (Schwarzschild metric).
    
    Parameters:
    - gravitational_potential (float): Gravitational potential in dimensionless units (range [0, 0.5]).

    Returns:
    - Gravitational time dilation factor.
    """
    return tf.sqrt(1 - 2 * gravitational_potential)

# ===========================================
# Loss Function to Minimize Differences Between Tick Rates
# ===========================================

def loss_function(quantum_rate, kinetic_rate, gravitational_rate):
    """
    Calculate the loss between quantum, kinetic, and gravitational time dilation rates.
    
    The objective is to minimize the squared difference between quantum time dilation
    and both kinetic and gravitational time dilation.

    Parameters:
    - quantum_rate (tensor): Quantum time dilation rate.
    - kinetic_rate (tensor): Kinetic time dilation rate (from velocity).
    - gravitational_rate (tensor): Gravitational time dilation rate (from gravitational potential).

    Returns:
    - Loss value (squared differences).
    """
    return tf.square(quantum_rate - kinetic_rate) + tf.square(quantum_rate - gravitational_rate)

# ===========================================
# Parameter Optimization Using Gradient Descent
# ===========================================

def optimize_parameters(initial_conditions, num_steps=1000, learning_rate=0.01):
    """
    Perform gradient descent to optimize parameters and minimize the time dilation loss.
    
    Parameters:
    - initial_conditions (dict): Initial values for monogamy degree, interaction strength, velocity, and gravitational potential.
    - num_steps (int): Number of optimization steps.
    - learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
    - loss_vals (list): List of loss values over time.
    - param_vals (dict): Dictionary containing parameter values over time.
    - rate_vals (dict): Dictionary containing quantum, kinetic, and gravitational time dilation rates over time.
    """
    # Adam optimizer for gradient-based optimization
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Define variables for each parameter with their initial values
    monogamy_degree = tf.Variable(initial_conditions["monogamy_degree"], name="monogamy_degree")
    interaction_strength = tf.Variable(initial_conditions["interaction_strength"], name="interaction_strength")
    velocity = tf.Variable(initial_conditions["velocity"], name="velocity")
    gravitational_potential = tf.Variable(initial_conditions["gravitational_potential"], name="gravitational_potential")

    # Store the loss and parameter values over time
    loss_vals = []
    param_vals = {"monogamy": [], "interaction": [], "velocity": [], "gravitational": []}
    rate_vals = {"quantum": [], "kinetic": [], "gravitational": []}

    # Optimization loop over the specified number of steps
    for _ in range(num_steps):
        # Record gradients and compute loss in the optimization context
        with tf.GradientTape() as tape:
            # Compute time dilation rates
            quantum_rate = quantum_tick_rate(monogamy_degree, interaction_strength)
            kinetic_rate = kinetic_tick_rate(velocity)
            gravitational_rate = gravitational_tick_rate(gravitational_potential)
            
            # Compute the loss function based on the differences between the rates
            loss = loss_function(quantum_rate, kinetic_rate, gravitational_rate)

        # Compute the gradients of the loss function with respect to each variable
        gradients = tape.gradient(loss, [monogamy_degree, interaction_strength, velocity, gravitational_potential])
        
        # Apply the gradients to update the variables
        optimizer.apply_gradients(zip(gradients, [monogamy_degree, interaction_strength, velocity, gravitational_potential]))

        # Apply constraints to the parameter ranges (clipping to valid physical ranges)
        monogamy_degree.assign(tf.clip_by_value(monogamy_degree, 0, 1))
        interaction_strength.assign(tf.clip_by_value(interaction_strength, 0, 1))
        velocity.assign(tf.clip_by_value(velocity, 0, 0.99))
        gravitational_potential.assign(tf.clip_by_value(gravitational_potential, 0, 0.5))

        # Record the values at each step
        loss_vals.append(loss.numpy())
        param_vals["monogamy"].append(monogamy_degree.numpy())
        param_vals["interaction"].append(interaction_strength.numpy())
        param_vals["velocity"].append(velocity.numpy())
        param_vals["gravitational"].append(gravitational_potential.numpy())
        rate_vals["quantum"].append(quantum_rate.numpy())
        rate_vals["kinetic"].append(kinetic_rate.numpy())
        rate_vals["gravitational"].append(gravitational_rate.numpy())

    return loss_vals, param_vals, rate_vals

# ===========================================
# Parameter Space Exploration for Quantum vs Relativistic Time Dilation
# ===========================================

def parameter_space_exploration(num_points=5, num_steps=1000):
    """
    Explore the parameter space by sampling monogamy degrees, interaction strengths, velocities, and gravitational potentials.
    
    This function performs optimization for different combinations of parameters and collects the results.

    Parameters:
    - num_points (int): Number of sample points in each parameter dimension.
    - num_steps (int): Number of optimization steps for each parameter combination.

    Returns:
    - results (list): List of tuples containing initial conditions, loss values, parameter values, and rate values.
    """
    # Define the range of values to explore for each parameter
    monogamy_range = np.linspace(0, 1, num_points)
    interaction_range = np.linspace(0, 1, num_points)
    velocity_range = np.linspace(0, 0.99, num_points)
    gravitational_range = np.linspace(0, 0.5, num_points)

    # Generate all combinations of parameter values
    parameter_combinations = list(product(monogamy_range, interaction_range, velocity_range, gravitational_range))
    results = []

    # Iterate through all parameter combinations
    for params in tqdm(parameter_combinations, desc="Exploring Parameter Space"):
        initial_conditions = {
            "monogamy_degree": params[0],
            "interaction_strength": params[1],
            "velocity": params[2],
            "gravitational_potential": params[3]
        }
        # Optimize for each combination and collect results
        loss_vals, param_vals, rate_vals = optimize_parameters(initial_conditions, num_steps)
        results.append((initial_conditions, loss_vals, param_vals, rate_vals))

    return results

# ===========================================
# Plotting the Results of the Optimization
# ===========================================

def plot_results(results):
    """
    Plot the results of the parameter space exploration, including time dilation factors and parameter evolution.
    
    Parameters:
    - results (list): List of optimization results (losses, parameters, and time dilation rates).
    """
    # Set up the figure and subplots
    fig, axs = plt.subplots(2, 3, figsize=(20, 20))
    fig.suptitle('Quantum and Relativistic Time Dilation Comparison (Parameter Space Exploration)')

    # Iterate through each result set and plot the data
    for initial_conditions, loss_vals, param_vals, rate_vals in results:
        steps = range(len(loss_vals))
        alpha = 0.1  # Transparency for the lines due to the number of plots

        # Plot time dilation factors over the optimization steps
        axs[0, 0].plot(steps, rate_vals['quantum'], 'b-', alpha=alpha)
        axs[0, 0].plot(steps, rate_vals['kinetic'], 'r-', alpha=alpha)
        axs[0, 0].plot(steps, rate_vals['gravitational'], 'g-', alpha=alpha)

        # Plot the evolution of parameters over time
        axs[0, 1].plot(steps, param_vals['monogamy'], 'b-', alpha=alpha)
        axs[0, 1].plot(steps, param_vals['interaction'], 'r-', alpha=alpha)
        axs[0, 1].plot(steps, param_vals['velocity'], 'g-', alpha=alpha)
        axs[0, 1].plot(steps, param_vals['gravitational'], 'y-', alpha=alpha)

        # Plot quantum vs gravitational time dilation
        axs[0, 2].scatter(rate_vals['quantum'][-1], rate_vals['gravitational'][-1], alpha=alpha)

    # Plot the relationship between entanglement strength and gravitational potential
    entanglement = [param_vals['interaction'][-1] for _, _, param_vals, _ in results]
    gravitational = [param_vals['gravitational'][-1] for _, _, param_vals, _ in results]
    axs[1, 0].hexbin(entanglement, gravitational, gridsize=20, cmap='viridis')

    # Plot the distribution of quantum time dilation factors
    final_quantum_rates = [rate_vals['quantum'][-1] for _, _, _, rate_vals in results]
    sns.histplot(final_quantum_rates, kde=True, ax=axs[1, 1])

    # Plot the distribution of correlations between entanglement and gravitational potential
    correlations = [np.corrcoef(param_vals['interaction'], param_vals['gravitational'])[0, 1] for _, _, param_vals, _ in results]
    sns.histplot(correlations, kde=True, ax=axs[1, 2])

    # Set titles and labels for all subplots
    axs[0, 0].set_title('Time Dilation Factors')
    axs[0, 0].set_xlabel('Optimization Steps')
    axs[0, 0].set_ylabel('Time Dilation Factor')

    axs[0, 1].set_title('Parameter Evolution')
    axs[0, 1].set_xlabel('Optimization Steps')
    axs[0, 1].set_ylabel('Parameter Value')

    axs[0, 2].set_title('Quantum vs Gravitational Time Dilation')
    axs[0, 2].set_xlabel('Quantum Time Dilation Factor')
    axs[0, 2].set_ylabel('Gravitational Time Dilation Factor')

    axs[1, 0].set_title('Entanglement Strength vs Gravitational Potential')
    axs[1, 0].set_xlabel('Entanglement Strength')
    axs[1, 0].set_ylabel('Gravitational Potential')

    axs[1, 1].set_title('Distribution of Final Quantum Time Dilation Factors')
    axs[1, 1].set_xlabel('Quantum Time Dilation Factor')
    axs[1, 1].set_ylabel('Frequency')

    axs[1, 2].set_title('Distribution of Correlations')
    axs[1, 2].set_xlabel('Correlation Coefficient')
    axs[1, 2].set_ylabel('Frequency')

    # Show the figure
    plt.tight_layout()
    plt.show()

# ===========================================
# Statistical Analysis of Results
# ===========================================

def analyze_results(results):
    """
    Perform enhanced statistical analysis on the results of the parameter space exploration.
    
    Parameters:
    - results (list): List of optimization results (losses, parameters, and time dilation rates).
    """
    final_values = {
        'quantum_rate': [],
        'gravitational_rate': [],
        'entanglement': [],
        'gravitational_potential': []
    }
    correlations = []

    # Extract final values for each key parameter and calculate correlations
    for _, _, param_vals, rate_vals in results:
        final_values['quantum_rate'].append(rate_vals['quantum'][-1])
        final_values['gravitational_rate'].append(rate_vals['gravitational'][-1])
        final_values['entanglement'].append(param_vals['interaction'][-1])
        final_values['gravitational_potential'].append(param_vals['gravitational'][-1])

        # Compute correlation if valid data is present
        if not np.isnan(param_vals['interaction'][-1]) and not np.isnan(param_vals['gravitational'][-1]):
            corr = np.corrcoef(param_vals['interaction'], param_vals['gravitational'])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

    # Descriptive statistics for final values
    print("\nEnhanced Statistical Analysis:")
    for key, values in final_values.items():
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            print(f"\n{key.replace('_', ' ').title()}:")
            print(f"  Mean: {np.mean(valid_values):.4f}")
            print(f"  Median: {np.median(valid_values):.4f}")
            print(f"  Standard Deviation: {np.std(valid_values):.4f}")
            print(f"  Range: {np.min(valid_values):.4f} to {np.max(valid_values):.4f}")
            print(f"  Number of valid values: {len(valid_values)}")

    # Perform correlation and regression analysis
    if correlations:
        print("\nCorrelation Analysis:")
        print(f"  Mean correlation: {np.mean(correlations):.4f}")
        print(f"  Median correlation: {np.median(correlations):.4f}")
        print(f"  Standard deviation of correlations: {np.std(correlations):.4f}")

        t_stat, p_value = stats.ttest_1samp(correlations, 0)
        print(f"\nOne-sample t-test (H0: mean correlation = 0):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")

    print("\nNote: These results are based on the available valid data points. Increase sample points for more robust analysis.")

# ===========================================
# Main Function to Execute the Simulation
# ===========================================

def main():
    """
    Main function to run the quantum time dilation simulation and perform analysis.
    """
    num_points = 6  # Number of points to sample in each parameter dimension
    num_steps = 1000  # Number of optimization steps for each parameter combination
    results = parameter_space_exploration(num_points, num_steps)
    
    # Plot and analyze results
    plot_results(results)
    analyze_results(results)

# Execute the main function when the script is run
if __name__ == "__main__":
    main()
