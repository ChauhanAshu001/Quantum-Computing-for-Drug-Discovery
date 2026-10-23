import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import matplotlib.pyplot as plt
from src.molecular_hamiltonian import get_molecular_problem, get_mapper
from src.ansatz_factory import build_hardware_efficient_ansatz
from src.solvers import solve_exact_classical, solve_vqe_quantum
from qiskit_algorithms.optimizers import COBYLA, SLSQP

os.makedirs("results/figures", exist_ok=True)
problem = get_molecular_problem(0.735)
mapper = get_mapper()
ansatz = build_hardware_efficient_ansatz()
exact_energy = solve_exact_classical(problem, mapper)

optimizers = {"COBYLA": COBYLA(maxiter=40), "SLSQP": SLSQP(maxiter=40)}
logs = {}

print("Running optimizer comparison...")
for name, opt in optimizers.items():
    history = []
    solve_vqe_quantum(problem, mapper, ansatz, opt, callback=lambda count, params, val, meta: history.append(val))
    logs[name] = history

plt.figure(figsize=(7, 4.5))
for name, hist in logs.items():
    plt.plot(range(1, len(hist) + 1), hist, marker="o", markersize=3, label=name)
plt.axhline(y=exact_energy, color="r", linestyle="--", label="Exact Baseline")
plt.title("Optimizer Convergence Landscape on VQE")
plt.xlabel("Iteration")
plt.ylabel("Energy (Hartree)")
plt.grid(True, linestyle=":")
plt.legend()
plt.tight_layout()
plt.savefig("results/figures/optimizer_convergence.png", dpi=300)
print("Finished! Saved optimizer_convergence.png")

# --- Hardware Resource Profiling ---
decomposed_ansatz = ansatz.decompose()
specs_content = f"""=== Quantum Circuit Specifications (NISQ Profile) ===
Target Molecule: H2 (Active Space, Bond Distance: 0.735 Angstrom)
Active Qubits: {ansatz.num_qubits}
Parameterized Variational Angles (theta): {ansatz.num_parameters}
Transpiled Circuit Depth: {decomposed_ansatz.depth()}
Gate Operation Counts: {dict(decomposed_ansatz.count_ops())}
Fermion-to-Qubit Mapping: Jordan-Wigner
Ansatz Type: Hardware-Efficient TwoLocal (Ry rotations, CZ entanglement)
"""

with open("results/circuit_specs.txt", "w") as f:
    f.write(specs_content)

print("Saved circuit specifications to results/circuit_specs.txt")