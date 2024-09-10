import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant
c = 299792458    # Speed of light
M = 1.989e30     # Mass of the Sun

# Range of r/r_s values (r_s is the Schwarzschild radius)
r_over_rs = np.logspace(0, 4, 1000)

# Calculate classical time dilation
r = r_over_rs * (2*G*M/c**2)
gamma_g = np.sqrt(1 - 2*G*M/(r*c**2))

# Function for higher-order corrections for quantum time dilation
def quantum_time_dilation(lambda_q):
    # Add higher-order corrections based on entanglement structure (from Section 5)
    alpha = 0.5  # Second-order correction coefficient
    beta = 0.1   # Third-order correction coefficient
    return np.exp(-lambda_q) * (1 + alpha * lambda_q**2 + beta * lambda_q**3)

# Dynamically adjust k based on the entanglement structure
def calculate_lambda_q(r, M, G, c, k_base=1, entanglement_strength=1):
    k_dynamic = k_base * (1 + 0.1 * entanglement_strength)  # Adjust k with entanglement
    return k_dynamic * G*M/(r*c**2)

# Simulate different entanglement strengths and their impact on quantum time dilation
entanglement_strengths = [0.5, 1, 2]  # Different levels of multipartite entanglement
colors = ['blue', 'green', 'red']

plt.figure(figsize=(12, 8))

# Plot classical time dilation
plt.semilogx(r_over_rs, gamma_g, label='Classical (GR)', linewidth=2)

# Loop over different entanglement strengths
for idx, entanglement_strength in enumerate(entanglement_strengths):
    lambda_q = calculate_lambda_q(r, M, G, c, entanglement_strength=entanglement_strength)
    gamma_q = quantum_time_dilation(lambda_q)
    plt.semilogx(r_over_rs, gamma_q, label=f'Quantum (Entanglement Strength {entanglement_strength})',
                 linewidth=2, linestyle='--', color=colors[idx])

plt.xlabel('r / r_s (Schwarzschild radii)')
plt.ylabel('Time Dilation Factor gamma')
plt.title('Comparison of Classical and Quantum Time Dilation with Higher-Order Corrections')
plt.legend()
plt.grid(True)
plt.ylim(0, 1.1)
plt.axhline(y=1, color='k', linestyle=':', alpha=0.5)
plt.text(1.1, 0.02, 'Event Horizon', rotation=90, verticalalignment='bottom')
plt.axvline(x=1, color='k', linestyle=':', alpha=0.5)

# Calculate and plot the difference between classical and quantum time dilation
plt.twinx()
for idx, entanglement_strength in enumerate(entanglement_strengths):
    lambda_q = calculate_lambda_q(r, M, G, c, entanglement_strength=entanglement_strength)
    gamma_q = quantum_time_dilation(lambda_q)
    diff = np.abs(gamma_g - gamma_q)
    plt.semilogx(r_over_rs, diff, label=f'|Difference| (Entanglement Strength {entanglement_strength})',
                 color=colors[idx], alpha=0.5)

plt.ylabel('Absolute Difference |gamma_g - gamma_q|')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()

# Print key values for quantum and classical time dilation at different points
for strength in entanglement_strengths:
    lambda_q = calculate_lambda_q(r, M, G, c, entanglement_strength=strength)
    gamma_q = quantum_time_dilation(lambda_q)
    print(f"Entanglement Strength {strength}:")
    print(f"At r = 2r_s: gamma_g = {gamma_g[np.argmin(np.abs(r_over_rs - 2))]:.4f}, gamma_q = {gamma_q[np.argmin(np.abs(r_over_rs - 2))]:.4f}")
    print(f"At r = 10r_s: gamma_g = {gamma_g[np.argmin(np.abs(r_over_rs - 10))]:.4f}, gamma_q = {gamma_q[np.argmin(np.abs(r_over_rs - 10))]:.4f}")
    print(f"At r = 100r_s: gamma_g = {gamma_g[np.argmin(np.abs(r_over_rs - 100))]:.4f}, gamma_q = {gamma_q[np.argmin(np.abs(r_over_rs - 100))]:.4f}")
    print()

