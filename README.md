# Quantum-Computing-for-Drug-Discovery

### Molecular Ground-State Energy & Active-Space Evaluation via VQE
**Reference Paper:** *Quantum-machine-assisted drug discovery (npj Drug Discovery, 2026)*

---

## 📌 Project Overview
Classical Computer-Aided Drug Design (CADD) relies on empirical, fixed-charge force fields that struggle to accurately capture dynamic electron correlation and polarization in critical enzyme/metal binding active sites. 

This project implements a hybrid quantum-classical pipeline using **Qiskit Nature**:
- **Classical Screening:** Fast filtering and geometric conformation of drug-like candidates.
- **Quantum Evaluator (VQE):** Targeted calculation of the molecular ground-state energy profile and binding well depth on the active-space electronic degrees of freedom.

We benchmark the **Variational Quantum Eigensolver (VQE)** against exact classical matrix diagonalization (`NumPyMinimumEigensolver`) and evaluate optimizer performance on the quantum energy landscape.

---

## 🏗️ Repository Architecture

```text
Quantum-Computing-for-Drug-Discovery/
├── data/
│   └── simulation_metrics.csv         # Computed Hartree energies and residual errors
├── env/                               # Local Python virtual environment
├── experiments/
│   ├── run_dissociation_sweep.py      # Potential energy dissociation curve (distance sweep)
│   └── run_optimizer_benchmark.py     # Optimizer convergence & circuit resource profiling
├── notebooks/
│   └── 01_qiskit_nature_reference.ipynb # Base Qiskit Nature tutorial notebook
├── results/
│   ├── figures/
│   │   ├── dissociation_curve.png     # Ground-state energy dissociation plot
│   │   └── optimizer_convergence.png  # COBYLA vs SLSQP convergence trajectory
│   └── circuit_specs.txt              # Circuit depth, qubit count, parameter count
├── src/
│   ├── molecular_hamiltonian.py       # PySCF molecular driver & Jordan-Wigner mapper
│   ├── ansatz_factory.py              # Parameterized TwoLocal ansatz construction
│   └── solvers.py                     # Classical exact & VQE solver execution interfaces
├── .gitignore
├── requirements.txt                   # Locked dependencies
└── README.md
```

---

## ⚙️ Methodology & Pipeline

1. **Second Quantization:** Define molecular active sites using `PySCFDriver` under the minimal `STO-3G` basis set.
2. **Fermion-to-Qubit Mapping:** Transform electronic creation/annihilation operators into Pauli spin operators (X, Y, Z) via the **Jordan-Wigner transformation**.
3. **Ansatz Circuit:** Construct a hardware-efficient `TwoLocal` variational circuit with Ry single-qubit rotations and CZ entanglement blocks.
4. **Hybrid Variational Optimization:** Measure the energy expectation value <H> using Qiskit's `StatevectorEstimator` and iteratively update circuit parameters θ using classical optimizers until ground-state energy converges.
5. **Algorithmic Evaluation:** Benchmark gradient-free (`COBYLA`) vs. gradient-based (`SLSQP`) optimizers and profile circuit complexity against NISQ hardware limits.

---

## 🚀 Setup & Execution

### 1. Environment Setup
```bash
# Clone the repository
git clone [https://github.com/ChauhanAshu001/Quantum-Computing-for-Drug-Discovery.git](https://github.com/ChauhanAshu001/Quantum-Computing-for-Drug-Discovery.git)
cd Quantum-Computing-for-Drug-Discovery

# Activate the virtual environment
# Windows (CMD / PowerShell):
env\Scripts\activate
# Linux / macOS:
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Experiments

```bash
# Generate Potential Energy Dissociation Curve:
python experiments/run_dissociation_sweep.py
# Outputs: results/figures/dissociation_curve.png and data/simulation_metrics.csv

# Run Optimizer Convergence & Hardware Resource Profiling:
python experiments/run_optimizer_benchmark.py
# Outputs: results/figures/optimizer_convergence.png and resource statistics in console.
```

---

## 📊 Key Results & Findings

- **Equilibrium Bond Length:** Verified at 0.735 Å with a minimum ground-state energy of -1.1373 Hartree.
- **Simulation Accuracy:** VQE ground-state energy matches exact classical matrix diagonalization with minimal residual error (< 1e-5 Hartree).
- **Optimizer Trade-Off:** `SLSQP` exhibits rapid quadratic convergence on noiseless mathematical surfaces, while `COBYLA` displays stability against gradient fluctuations.
- **Hardware Profile:** Requires 4 active qubits, 12 variational parameters (θ), and low two-qubit gate depth, validating its feasibility for near-term NISQ simulation.

---

## 📚 References
- Zhou et al., *"Quantum-machine-assisted drug discovery"*, npj Drug Discovery, 2026.
- Qiskit Nature Documentation & Tutorials: `03_ground_state_solvers`.