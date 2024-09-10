import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
G = 1  # Gravitational constant (in geometric units)
c = 1  # Speed of light
M = 1  # Black hole mass
hbar = 1  # Reduced Planck constant

# Simulation parameters
r_range = np.linspace(2.001*G*M/c**2, 20*G*M/c**2, 2000)  # Extended Radial distance range
lambda_max = 1  # Maximum entanglement parameter

# Quantum curvature tensor function
def quantum_curvature_tensor(r, lambda_r):
    """Compute the quantum curvature tensor inspired by the Ricci tensor."""
    classical_curvature = classical_curvature_kretschmann(r)
    quantum_correction = 8 * np.pi * G * quantum_stress_energy_tensor(r) / c**4
    return classical_curvature + quantum_correction

def safe_divide(a, b, fill_value=0):
    """Safely divide two values or arrays, returning fill_value where division by zero would occur."""
    if np.isscalar(a) and np.isscalar(b):
        return a / b if b != 0 else fill_value
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(a, b)
            if isinstance(c, np.ndarray):
                c[~np.isfinite(c)] = fill_value
            elif not np.isfinite(c):
                c = fill_value
            return c

def schwarzschild_metric(r):
    """Compute the Schwarzschild metric component g_tt."""
    return 1 - 2*G*M/(r*c**2)

def classical_curvature_kretschmann(r):
    """Compute the classical spacetime curvature (Kretschmann scalar)."""
    return safe_divide(48*(G*M)**2, (c**4 * r**6))

def quantum_fisher_information(r):
    """Compute the quantum Fisher information as a function of radius."""
    exponent = -2*lambda_max * (r - 2*G*M/c**2) / (2*G*M/c**2)
    denominator = 1 - np.exp(np.clip(exponent, -700, 700))  # Prevent overflow
    return safe_divide(1, denominator)

def entanglement_monogamy_parameter(r):
    """Compute the entanglement monogamy parameter λ(r) using concurrence."""
    return lambda_max * np.clip(1 - (r - 2*G*M/c**2) / (2*G*M/c**2), 0, 1)

def quantum_stress_energy_tensor(r):
    """Compute the quantum stress-energy tensor T_μν^ent."""
    lambda_r = entanglement_monogamy_parameter(r)
    return safe_divide(hbar * lambda_r**2, (8*np.pi*G*r**3))

def quantum_corrected_curvature(r):
    """Compute the quantum-corrected spacetime curvature."""
    classical = classical_curvature_kretschmann(r)
    quantum_correction = quantum_stress_energy_tensor(r)
    return classical + 8*np.pi*G*quantum_correction / c**4

def quantum_correction_geodesic(r):
    """Compute the quantum correction to the geodesic equation."""
    return safe_divide(quantum_corrected_curvature(r) - classical_curvature_kretschmann(r), 8*np.pi*G/c**4)

def geodesic_equation(t, y, is_quantum):
    """Geodesic equation for radial motion."""
    r, dr_dt = y
    correction = quantum_correction_geodesic(r) if is_quantum else 0
    d2r_dt2 = -G*M/r**2 + (3*G*M*dr_dt**2)/(c**2*r) - correction
    return [dr_dt, d2r_dt2]

# Compute metrics
classical_curv = classical_curvature_kretschmann(r_range)
quantum_curv = quantum_corrected_curvature(r_range)
fisher_info = quantum_fisher_information(r_range)

# Solve geodesic equations
t_span = (0, 100)
y0 = [5*G*M/c**2, 0]  # Initial conditions: r0 = 5Rs, dr/dt = 0

# Classical geodesic
sol_classical = solve_ivp(geodesic_equation, t_span, y0, args=(False,), dense_output=True)

# Quantum-corrected geodesic
sol_quantum = solve_ivp(geodesic_equation, t_span, y0, args=(True,), dense_output=True)

# Generate solution points
t_eval = np.linspace(0, 100, 1000)
sol_classical_points = sol_classical.sol(t_eval)
sol_quantum_points = sol_quantum.sol(t_eval)

# Plotting
plt.figure(figsize=(15, 10))

# Curvature comparison
plt.subplot(2, 2, 1)
plt.plot(r_range, classical_curv, label='Classical')
plt.plot(r_range, quantum_curv, label='Quantum-corrected')
plt.xlabel('Radial distance (r)')
plt.ylabel('Curvature')
plt.title('Spacetime Curvature (Quantum-Corrected vs Classical)')
plt.legend()
plt.yscale('log')

# Quantum Fisher Information
plt.subplot(2, 2, 2)
plt.plot(r_range, fisher_info)
plt.xlabel('Radial distance (r)')
plt.ylabel('Quantum Fisher Information')
plt.title('Quantum Fisher Information vs. Radius')

# Geodesic comparison
plt.subplot(2, 2, 3)
plt.plot(t_eval, sol_classical_points[0], label='Classical')
plt.plot(t_eval, sol_quantum_points[0], label='Quantum-corrected')
plt.xlabel('Proper time')
plt.ylabel('Radial distance (r)')
plt.title('Geodesic Comparison (Quantum-Corrected vs Classical)')
plt.legend()

# Entanglement monogamy parameter
plt.subplot(2, 2, 4)
lambda_r = entanglement_monogamy_parameter(r_range)
plt.plot(r_range, lambda_r)
plt.xlabel('Radial distance (r)')
plt.ylabel('λ(r)')
plt.title('Entanglement Monogamy Parameter λ(r)')

plt.tight_layout()
plt.show()

# 3D visualization of quantum curvature tensor (for intuition)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
R, L = np.meshgrid(r_range, lambda_r)
quantum_curv_tensor = quantum_curvature_tensor(R, L)
ax.plot_surface(R, L, quantum_curv_tensor, cmap='viridis')
ax.set_xlabel('Radial distance (r)')
ax.set_ylabel('Monogamy Parameter λ(r)')
ax.set_zlabel('Quantum Curvature Tensor')
ax.set_title('3D Visualization of Quantum Curvature Tensor')
plt.show()

# Print key results
horizon_index = np.argmin(np.abs(r_range - 2*G*M/c**2))
print(f"Classical curvature at horizon: {classical_curv[horizon_index]:.2e}")
print(f"Quantum-corrected curvature at horizon: {quantum_curv[horizon_index]:.2e}")
print(f"Quantum Fisher Information peak: {np.max(fisher_info):.2f}")
print(f"Radial distance of QFI peak: {r_range[np.argmax(fisher_info)]:.2f}")
