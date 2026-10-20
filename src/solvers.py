from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit.primitives import StatevectorEstimator

def solve_exact_classical(problem, mapper):
    """Finds exact ground-state energy via classical matrix diagonalization."""
    classical_solver = NumPyMinimumEigensolver()
    calc = GroundStateEigensolver(mapper, classical_solver)
    res = calc.solve(problem)
    return res.total_energies[0]

def solve_vqe_quantum(problem, mapper, ansatz, optimizer, callback=None):
    """Finds ground-state energy using the hybrid VQE pipeline."""
    vqe = VQE(
        estimator=StatevectorEstimator(),
        ansatz=ansatz,
        optimizer=optimizer,
        callback=callback
    )
    calc = GroundStateEigensolver(mapper, vqe)
    res = calc.solve(problem)
    return res.total_energies[0]