import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, tensor, sigmaz, sigmax, basis, mesolve, destroy, fidelity, entropy_vn

class QuantumSpacetimeSimulation:
    def __init__(self, num_qubits, entanglement_strength, decoherence_rate=0.0, initial_state_type='ground'):
        self.num_qubits = num_qubits
        self.entanglement_strength = entanglement_strength
        self.decoherence_rate = decoherence_rate
        self.initial_state_type = initial_state_type  # 'ground', 'superposition', 'entangled'
        self.state = self.initialize_state()

    def initialize_state(self):
        if self.initial_state_type == 'ground':
            return tensor([basis(2, 0) for _ in range(self.num_qubits)])
        elif self.initial_state_type == 'superposition':
            return tensor([(basis(2, 0) + basis(2, 1)).unit() for _ in range(self.num_qubits)])
        elif self.initial_state_type == 'entangled':
            half_num = self.num_qubits // 2
            entangled_pairs = tensor([basis(2, 0), basis(2, 0)]) + tensor([basis(2, 1), basis(2, 1)])
            for _ in range(half_num - 1):
                entangled_pairs = tensor(entangled_pairs, tensor([basis(2, 0), basis(2, 0)]) + tensor([basis(2, 1), basis(2, 1)]))
            if self.num_qubits % 2 == 1:
                entangled_pairs = tensor(entangled_pairs, basis(2, 0))
            return entangled_pairs.unit()
        else:
            raise ValueError("Invalid initial state type.")

    def create_hamiltonian(self):
        """Create the system's Hamiltonian with local and interaction terms."""
        H = 0
        for i in range(self.num_qubits):
            h_i = tensor([sigmaz() if k == i else Qobj(np.eye(2)) for k in range(self.num_qubits)])
            H += h_i
        for i in range(self.num_qubits - 1):
            interaction_strength = self.entanglement_strength if i % 2 == 0 else self.entanglement_strength / 2  # Non-uniform entanglement
            h_ij = tensor([sigmax() if k in (i, i + 1) else Qobj(np.eye(2)) for k in range(self.num_qubits)])
            H += interaction_strength * h_ij
        return H

    def create_decoherence_operators(self):
        """Create Lindblad operators for decoherence effects."""
        c_ops = []
        for i in range(self.num_qubits):
            op = tensor([destroy(2) if k == i else Qobj(np.eye(2)) for k in range(self.num_qubits)])
            c_ops.append(np.sqrt(self.decoherence_rate) * op)
        return c_ops

    def time_evolution(self, t_max, dt):
        """Evolve the system state over time with possible decoherence."""
        H = self.create_hamiltonian()
        c_ops = self.create_decoherence_operators() if self.decoherence_rate > 0 else []
        tlist = np.arange(0, t_max, dt)
        result = mesolve(H, self.state, tlist, c_ops, [])
        return tlist, result.states

    def measure_entanglement(self, state):
        """Measure the mean von Neumann entropy across all qubits."""
        rho = state * state.dag()
        return np.mean([entropy_vn(rho.ptrace([i])) for i in range(self.num_qubits)])

    def measure_fubini_study_metric(self, state, prev_state):
        """Calculate the Fubini-Study metric between two consecutive states."""
        overlap = fidelity(state, prev_state)
        return np.arccos(np.sqrt(overlap)) ** 2

    def extract_metric_tensor(self, metric_value, dt):
        """Extract a 1D metric tensor from a scalar metric value."""
        return np.array([[metric_value / (dt ** 2)]])

    def calculate_curvature(self, metric_tensor):
        """Calculate the Ricci scalar curvature from the metric tensor."""
        det_g = np.linalg.det(metric_tensor)
        return 1 - det_g

    def run_simulation(self, t_max, dt):
        """Run the full simulation and collect all relevant observables."""
        times, states = self.time_evolution(t_max, dt)
        entanglements = [self.measure_entanglement(state) for state in states]
        fubini_study = [self.measure_fubini_study_metric(states[i], states[i - 1]) for i in range(1, len(states))]
        fubini_study.insert(0, 0)
        curvatures = [self.calculate_curvature(self.extract_metric_tensor(f, dt)) for f in fubini_study]

        return times, entanglements, fubini_study, curvatures

    def plot_results(self, times, entanglements, fubini_study, curvatures, gr_curvatures, label):
        """Plot the results for entanglement, Fubini-Study metric, curvature evolution, and compare with GR."""
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        axs[0, 0].plot(times, entanglements, label="Entanglement")
        axs[0, 0].set_xlabel("Time")
        axs[0, 0].set_ylabel("Von Neumann Entropy")
        axs[0, 0].set_title("Entanglement Evolution")
        axs[0, 0].legend()

        axs[0, 1].plot(times, fubini_study, label="Fubini-Study Metric")
        axs[0, 1].set_xlabel("Time")
        axs[0, 1].set_ylabel("Fubini-Study Metric")
        axs[0, 1].set_title("Geometric Evolution (Fubini-Study)")
        axs[0, 1].legend()

        axs[1, 0].plot(times, curvatures, label="Curvature")
        axs[1, 0].set_xlabel("Time")
        axs[1, 0].set_ylabel("Curvature")
        axs[1, 0].set_title("Curvature Evolution")
        axs[1, 0].legend()

        axs[1, 1].plot(times, curvatures, label="Quantum Curvature")
        axs[1, 1].plot(times, gr_curvatures, label="GR Predicted Curvature", linestyle="--")
        axs[1, 1].set_xlabel("Time")
        axs[1, 1].set_ylabel("Curvature")
        axs[1, 1].set_title("Curvature Comparison: Quantum vs GR")
        axs[1, 1].legend()

        plt.suptitle(f"Simulation Results - Initial State: {label}")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def comparative_analysis_with_gr(self, times, curvatures):
        """Placeholder function to simulate GR curvature for comparison."""
        gr_curvatures = np.linspace(curvatures[0], curvatures[-1], len(curvatures))  # Simulated GR curvature for comparison
        return gr_curvatures

# Example usage

# Diverse initial conditions
initial_conditions = ['ground', 'superposition', 'entangled']

# Running the simulations
for condition in initial_conditions:
    sim = QuantumSpacetimeSimulation(num_qubits=3, entanglement_strength=1.0, decoherence_rate=0.01, initial_state_type=condition)

    # Run the simulation
    times, entanglements, fubini_study, curvatures = sim.run_simulation(t_max=100, dt=0.1)

    # Generate GR comparison data
    gr_curvatures = sim.comparative_analysis_with_gr(times, curvatures)

    # Plot all results in a single image
    sim.plot_results(times, entanglements, fubini_study, curvatures, gr_curvatures, label=condition)
