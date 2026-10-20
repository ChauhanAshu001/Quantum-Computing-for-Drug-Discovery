from qiskit.circuit.library import TwoLocal

def build_hardware_efficient_ansatz(reps: int = 2):
    """
    Constructs parameterized TwoLocal ansatz with Ry rotations 
    and CZ entangling layers.
    """
    return TwoLocal(rotation_blocks="ry", entanglement_blocks="cz", reps=reps)git pull origin main