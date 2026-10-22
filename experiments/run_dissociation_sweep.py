import os
import pandas as pd
import matplotlib.pyplot as plt
from src.molecular_hamiltonian import get_molecular_problem, get_mapper
from src.ansatz_factory import build_hardware_efficient_ansatz
from src.solvers import solve_exact_classical, solve_vqe_quantum
from qiskit_algorithms.optimizers import SLSQP

os.makedirs("results/figures", exist_ok=True)
distances = [0.5, 0.735, 1.0, 1.5, 2.0]
records = []
mapper = get_mapper()
ansatz = build_hardware_efficient_ansatz()

print("Running bond distance sweep...")
for d in distances:
    problem = get_molecular_problem(d)
    exact_energy = solve_exact_classical(problem, mapper)
    vqe_energy = solve_vqe_quantum(problem, mapper, ansatz, SLSQP(maxiter=50))
    
    records.append({
        "Distance_A": d,
        "Exact_Energy_Ha": exact_energy,
        "VQE_Energy_Ha": vqe_energy,
        "Error_Ha": abs(vqe_energy - exact_energy)
    })
    print(f"Dist: {d:4.3f} Å | Exact: {exact_energy:.6f} Ha | VQE: {vqe_energy:.6f} Ha")

df = pd.DataFrame(records)
df.to_csv("results/simulation_metrics.csv", index=False)

plt.figure(figsize=(7, 4.5))
plt.plot(df["Distance_A"], df["Exact_Energy_Ha"], "k--", label="Exact Classical Benchmark")
plt.plot(df["Distance_A"], df["VQE_Energy_Ha"], "ro-", label="Quantum VQE Simulation")
plt.title("Potential Energy Dissociation Profile")
plt.xlabel("Distance (Å)")
plt.ylabel("Total Energy (Hartree)")
plt.grid(True, linestyle=":")
plt.legend()
plt.tight_layout()
plt.savefig("results/figures/dissociation_curve.png", dpi=300)
print("Finished! Saved dissociation_curve.png and simulation_metrics.csv")