import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
G = 1  # Gravitational constant
c = 1  # Speed of light
hbar = 1  # Reduced Planck constant
k_B = 1  # Boltzmann constant
M_planck = 1  # Planck mass (in geometrized units)


# Functions for black hole physics
def hawking_temperature(M):
    """Calculate Hawking temperature for a black hole of mass M."""
    return hbar * c ** 3 / (8 * np.pi * G * k_B * M)


def evaporation_rate(M):
    """Calculate mass loss rate due to Hawking radiation."""
    return -hbar * c ** 6 / (15360 * np.pi * G ** 2 * M ** 2)


def entanglement_entropy(M):
    """Calculate entanglement entropy between interior and exterior of the black hole."""
    return 4 * np.pi * G * M ** 2 / hbar


def quantum_metric_factor(M, M0):
    """Quantum correction factor for spacetime metric."""
    return 1 - np.exp(-M / M0)


def spacetime_emergence(M, M0):
    """Model emergent spacetime structure based on entanglement."""
    r = np.linspace(2 * G * M / c ** 2, 20 * G * M / c ** 2, 1000)  # Expanded radial distance range
    classical_metric = 1 - 2 * G * M / (r * c ** 2)
    quantum_metric = classical_metric * quantum_metric_factor(M, M0)
    return r, quantum_metric


def black_hole_evolution(t, y):
    M = y[0]
    dMdt = evaporation_rate(M)
    return [dMdt]


# Define different black hole masses: supermassive, solar, and Planck-scale black hole
masses = {
    "Supermassive": 1e9,  # Mass in geometric units
    "Solar-mass": 1e1,  # Mass in geometric units
    "Planck-scale": 1e-1  # Near Planck mass
}

t_span = (0, 1e6)  # Time span for evolution

# Set up the figure for the 3 cases
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

for idx, (bh_type, M0) in enumerate(masses.items()):
    # Solve black hole evolution for each case
    sol = solve_ivp(black_hole_evolution, t_span, [M0], dense_output=True)

    # Generate solution points
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    M_evolution = sol.sol(t_eval)[0]

    # Calculate derived quantities
    S_ent = entanglement_entropy(M_evolution)
    T_hawking = hawking_temperature(M_evolution)

    # Get spacetime structure
    r, quantum_metric = spacetime_emergence(M_evolution[-1], M0)

    # Plotting for each black hole type
    axes[idx, 0].plot(t_eval, M_evolution)
    axes[idx, 0].set_xlabel('Time')
    axes[idx, 0].set_ylabel('Black Hole Mass')
    axes[idx, 0].set_title(f'{bh_type} Black Hole Evaporation')

    axes[idx, 1].plot(t_eval, S_ent)
    axes[idx, 1].set_xlabel('Time')
    axes[idx, 1].set_ylabel('Entanglement Entropy')
    axes[idx, 1].set_title(f'{bh_type} Entanglement Entropy Evolution')

    axes[idx, 2].plot(t_eval, T_hawking)
    axes[idx, 2].set_xlabel('Time')
    axes[idx, 2].set_ylabel('Hawking Temperature')
    axes[idx, 2].set_title(f'{bh_type} Hawking Temperature Evolution')

    axes[idx, 3].plot(r, quantum_metric, label='Quantum-corrected')
    axes[idx, 3].plot(r, 1 - 2 * G * M_evolution[-1] / (r * c ** 2), '--', label='Classical')
    axes[idx, 3].set_xlabel('Radial Distance')
    axes[idx, 3].set_ylabel('Metric Factor')
    axes[idx, 3].set_title(f'{bh_type} Emergent Spacetime Structure')
    axes[idx, 3].legend()

plt.tight_layout()
plt.show()
