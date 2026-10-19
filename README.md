# Quantum-Machine-Assisted Drug Discovery: Active-Site Molecular Evaluation via VQE

An implementation and benchmarking study of the Variational Quantum Eigensolver (VQE) for molecular ground-state interaction modeling, based on the framework proposed in *Zhou et al., npj Drug Discovery (2026)*.

---

## 📌 Problem Motivation
Classical Computer-Aided Drug Design (CADD) methods rely on fixed-charge force fields that oversimplify dynamic electron correlation and polarization in critical enzyme/metal binding active sites. 

This project implements a hybrid quantum-classical pipeline: classical methods handle large-scale geometric screening, while a quantum processor (simulated via Qiskit) acts as a high-precision active-space evaluator to compute ground-state potential energy curves and binding interactions.

---

## 🚀 Key Features & Pipeline
- **Second Quantization Formulation:** Uses `PySCF` to construct molecular electronic Hamiltonians (STO-3G basis).
- **Fermion-to-Qubit Mapping:** Converts electronic creation/annihilation operators into Pauli spin operators via the **Jordan-Wigner transformation**.
- **Hardware-Efficient Parameterized Ansatz:** Utilizes a `TwoLocal` circuit with $R_Y$ rotation blocks and $CZ$ entanglement layers.
- **Hybrid Optimization Loop:** Executes the variational loop using the `StatevectorEstimator` primitive paired with classical numerical optimizers.
- **Empirical Benchmarking:** Compares convergence speed and stability between gradient-free (`COBYLA`) and gradient-based (`SLSQP`) optimizers against exact classical matrix diagonalization (`NumPyMinimumEigensolver`).

---

## 📂 Repository Structure
- `src/`: Modular code components (Hamiltonian constructor, circuit ansatz, solver wrappers).
- `experiments/`: Executable scripts to run the distance sweep and optimizer benchmarks.
- `results/`: High-resolution figures and exported metrics ready for review.
- `notebooks/`: Exploratory Jupyter notebook adapted from the official Qiskit Nature framework.

---

## 🛠️ Setup & Execution

### 1. Installation
\`\`\`bash
git clone https://github.com/<your-username>/quantum-drug-discovery-vqe.git
cd quantum-drug-discovery-vqe
python -m venv venv

# Activate Virtual Environment:
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
\`\`\`

### 2. Run Experiments
\`\`\`bash
# Compute potential energy dissociation curve:
python experiments/run_dissociation_sweep.py

# Run optimizer comparison and resource profiling:
python experiments/run_optimizer_benchmark.py
\`\`\`

---

## 📊 Results Summary
- **Equilibrium Bond Length:** $0.735\text{ \AA}$ at $-1.1373\text{ Hartree}$.
- **Hardware Resource Profile:** 4 Active Qubits, 12 Trainable Parameters ($\theta$).
- **Optimizer Trade-off:** `SLSQP` exhibits faster asymptotic convergence on clean landscapes, whereas `COBYLA` demonstrates robustness in the presence of landscape fluctuations.