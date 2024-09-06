import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from scipy import stats
import seaborn as sns

def quantum_tick_rate(monogamy_degree, interaction_strength):
    return tf.exp(-monogamy_degree * interaction_strength)

def kinetic_tick_rate(velocity):
    return tf.sqrt(1 - tf.square(velocity))

def gravitational_tick_rate(gravitational_potential):
    return tf.sqrt(1 - 2 * gravitational_potential)

def loss_function(quantum_rate, kinetic_rate, gravitational_rate):
    return tf.square(quantum_rate - kinetic_rate) + tf.square(quantum_rate - gravitational_rate)

def optimize_parameters(initial_conditions, num_steps=1000, learning_rate=0.01):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    monogamy_degree = tf.Variable(initial_conditions["monogamy_degree"], name="monogamy_degree")
    interaction_strength = tf.Variable(initial_conditions["interaction_strength"], name="interaction_strength")
    velocity = tf.Variable(initial_conditions["velocity"], name="velocity")
    gravitational_potential = tf.Variable(initial_conditions["gravitational_potential"], name="gravitational_potential")

    loss_vals = []
    param_vals = {"monogamy": [], "interaction": [], "velocity": [], "gravitational": []}
    rate_vals = {"quantum": [], "kinetic": [], "gravitational": []}

    for _ in range(num_steps):
        with tf.GradientTape() as tape:
            quantum_rate = quantum_tick_rate(monogamy_degree, interaction_strength)
            kinetic_rate = kinetic_tick_rate(velocity)
            gravitational_rate = gravitational_tick_rate(gravitational_potential)
            loss = loss_function(quantum_rate, kinetic_rate, gravitational_rate)

        gradients = tape.gradient(loss, [monogamy_degree, interaction_strength, velocity, gravitational_potential])
        optimizer.apply_gradients(zip(gradients, [monogamy_degree, interaction_strength, velocity, gravitational_potential]))

        monogamy_degree.assign(tf.clip_by_value(monogamy_degree, 0, 1))
        interaction_strength.assign(tf.clip_by_value(interaction_strength, 0, 1))
        velocity.assign(tf.clip_by_value(velocity, 0, 0.99))
        gravitational_potential.assign(tf.clip_by_value(gravitational_potential, 0, 0.5))

        loss_vals.append(loss.numpy())
        param_vals["monogamy"].append(monogamy_degree.numpy())
        param_vals["interaction"].append(interaction_strength.numpy())
        param_vals["velocity"].append(velocity.numpy())
        param_vals["gravitational"].append(gravitational_potential.numpy())
        rate_vals["quantum"].append(quantum_rate.numpy())
        rate_vals["kinetic"].append(kinetic_rate.numpy())
        rate_vals["gravitational"].append(gravitational_rate.numpy())

    return loss_vals, param_vals, rate_vals

def parameter_space_exploration(num_points=5, num_steps=1000):
    monogamy_range = np.linspace(0, 1, num_points)
    interaction_range = np.linspace(0, 1, num_points)
    velocity_range = np.linspace(0, 0.99, num_points)
    gravitational_range = np.linspace(0, 0.5, num_points)

    parameter_combinations = list(product(monogamy_range, interaction_range, velocity_range, gravitational_range))
    results = []

    for params in tqdm(parameter_combinations, desc="Exploring Parameter Space"):
        initial_conditions = {
            "monogamy_degree": params[0],
            "interaction_strength": params[1],
            "velocity": params[2],
            "gravitational_potential": params[3]
        }
        loss_vals, param_vals, rate_vals = optimize_parameters(initial_conditions, num_steps)
        results.append((initial_conditions, loss_vals, param_vals, rate_vals))

    return results


def plot_results(results):
    fig, axs = plt.subplots(2, 3, figsize=(20, 20))
    fig.suptitle('Quantum and Relativistic Time Dilation Comparison (Parameter Space Exploration)')

    for initial_conditions, loss_vals, param_vals, rate_vals in results:
        steps = range(len(loss_vals))
        alpha = 0.1  # Make lines more transparent due to many runs

        # Plot 1: Time Dilation Factors
        axs[0, 0].plot(steps, rate_vals['quantum'], 'b-', alpha=alpha)
        axs[0, 0].plot(steps, rate_vals['kinetic'], 'r-', alpha=alpha)
        axs[0, 0].plot(steps, rate_vals['gravitational'], 'g-', alpha=alpha)

        # Plot 2: Parameter Evolution
        axs[0, 1].plot(steps, param_vals['monogamy'], 'b-', alpha=alpha)
        axs[0, 1].plot(steps, param_vals['interaction'], 'r-', alpha=alpha)
        axs[0, 1].plot(steps, param_vals['velocity'], 'g-', alpha=alpha)
        axs[0, 1].plot(steps, param_vals['gravitational'], 'y-', alpha=alpha)

        # Plot 3: Quantum vs Gravitational Time Dilation
        axs[0, 2].scatter(rate_vals['quantum'][-1], rate_vals['gravitational'][-1], alpha=alpha)

    # Plot 4: Entanglement Strength vs Gravitational Potential (Heatmap)
    entanglement = [param_vals['interaction'][-1] for _, _, param_vals, _ in results]
    gravitational = [param_vals['gravitational'][-1] for _, _, param_vals, _ in results]
    axs[1, 0].hexbin(entanglement, gravitational, gridsize=20, cmap='viridis')

    # Plot 5: Distribution of final quantum time dilation factors
    final_quantum_rates = [rate_vals['quantum'][-1] for _, _, _, rate_vals in results]
    sns.histplot(final_quantum_rates, kde=True, ax=axs[1, 1])

    # Plot 6: Distribution of correlations
    correlations = [np.corrcoef(param_vals['interaction'], param_vals['gravitational'])[0, 1]
                    for _, _, param_vals, _ in results]
    sns.histplot(correlations, kde=True, ax=axs[1, 2])

    axs[0, 0].set_title('Time Dilation Factors')
    axs[0, 0].set_xlabel('Optimization Steps')
    axs[0, 0].set_ylabel('Time Dilation Factor')
    axs[0, 0].legend(['Quantum', 'Kinetic', 'Gravitational'])

    axs[0, 1].set_title('Parameter Evolution')
    axs[0, 1].set_xlabel('Optimization Steps')
    axs[0, 1].set_ylabel('Parameter Value')
    axs[0, 1].legend(['Monogamy Degree', 'Interaction Strength', 'Velocity (c)', 'Grav. Potential (GM/rc²)'])

    axs[0, 2].plot([0, 1], [0, 1], 'r--')
    axs[0, 2].set_title('Quantum vs Gravitational Time Dilation')
    axs[0, 2].set_xlabel('Quantum Time Dilation Factor')
    axs[0, 2].set_ylabel('Gravitational Time Dilation Factor')

    axs[1, 0].set_title('Entanglement Strength vs Gravitational Potential')
    axs[1, 0].set_xlabel('Entanglement Strength')
    axs[1, 0].set_ylabel('Gravitational Potential (GM/rc²)')

    axs[1, 1].set_title('Distribution of Final Quantum Time Dilation Factors')
    axs[1, 1].set_xlabel('Quantum Time Dilation Factor')
    axs[1, 1].set_ylabel('Frequency')

    axs[1, 2].set_title('Distribution of Correlations')
    axs[1, 2].set_xlabel('Correlation Coefficient')
    axs[1, 2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def analyze_results(results):
    final_values = {
        'quantum_rate': [],
        'gravitational_rate': [],
        'entanglement': [],
        'gravitational_potential': []
    }
    correlations = []

    for _, _, param_vals, rate_vals in results:
        final_values['quantum_rate'].append(rate_vals['quantum'][-1])
        final_values['gravitational_rate'].append(rate_vals['gravitational'][-1])
        final_values['entanglement'].append(param_vals['interaction'][-1])
        final_values['gravitational_potential'].append(param_vals['gravitational'][-1])

        # Check for valid values before calculating correlation
        if not np.isnan(param_vals['interaction'][-1]) and not np.isnan(param_vals['gravitational'][-1]):
            corr = np.corrcoef(param_vals['interaction'], param_vals['gravitational'])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

    print("\nEnhanced Statistical Analysis:")

    # Descriptive statistics
    for key, values in final_values.items():
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            print(f"\n{key.replace('_', ' ').title()}:")
            print(f"  Mean: {np.mean(valid_values):.4f}")
            print(f"  Median: {np.median(valid_values):.4f}")
            print(f"  Standard Deviation: {np.std(valid_values):.4f}")
            print(f"  Range: {np.min(valid_values):.4f} to {np.max(valid_values):.4f}")
            print(f"  Number of valid values: {len(valid_values)}")
        else:
            print(f"\n{key.replace('_', ' ').title()}: No valid values")

    if correlations:
        print("\nCorrelation Analysis:")
        print(f"  Mean correlation: {np.mean(correlations):.4f}")
        print(f"  Median correlation: {np.median(correlations):.4f}")
        print(f"  Standard deviation of correlations: {np.std(correlations):.4f}")
        print(f"  Number of valid correlations: {len(correlations)}")

        # Hypothesis testing
        t_stat, p_value = stats.ttest_1samp(correlations, 0)
        print(f"\nOne-sample t-test (H0: mean correlation = 0):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")

        if p_value < 0.05:
            print("  The correlation between entanglement and gravitational potential is statistically significant.")
        else:
            print("  There is not enough evidence to conclude a significant correlation.")

        # Effect size
        cohen_d = (np.mean(correlations) - 0) / np.std(correlations)
        print(f"\nEffect size (Cohen's d): {cohen_d:.4f}")
        if abs(cohen_d) < 0.2:
            print("  The effect size is small.")
        elif abs(cohen_d) < 0.5:
            print("  The effect size is medium.")
        else:
            print("  The effect size is large.")

    # Regression analysis
    valid_entanglement = [v for v in final_values['entanglement'] if not np.isnan(v)]
    valid_gravitational = [v for v in final_values['gravitational_potential'] if not np.isnan(v)]

    if len(valid_entanglement) == len(valid_gravitational) and len(valid_entanglement) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(valid_entanglement, valid_gravitational)
        print("\nLinear Regression (Entanglement vs Gravitational Potential):")
        print(f"  Slope: {slope:.4f}")
        print(f"  Intercept: {intercept:.4f}")
        print(f"  R-squared: {r_value ** 2:.4f}")
        print(f"  P-value: {p_value:.4f}")

        if p_value < 0.05:
            print("  There is a significant linear relationship between entanglement and gravitational potential.")
        else:
            print("  There is not enough evidence to conclude a significant linear relationship.")
    else:
        print("\nNot enough valid data points for regression analysis.")

    print(
        "\nNote: These results are based on the available valid data points. Increase the number of sample points for more robust analysis.")


def main():
    num_points = 6  # Number of points to sample in each parameter dimension
    num_steps = 1000  # Number of optimization steps for each run
    results = parameter_space_exploration(num_points, num_steps)
    plot_results(results)
    analyze_results(results)


if __name__ == "__main__":
    main()
