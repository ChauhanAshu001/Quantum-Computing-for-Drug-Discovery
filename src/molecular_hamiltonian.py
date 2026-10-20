from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

def get_molecular_problem(distance: float = 0.735):
    """Initializes PySCF driver for H2 molecule at specified bond distance."""
    driver = PySCFDriver(
        atom=f"H 0 0 0; H 0 0 {distance}",
        basis="sto-3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM
    )
    return driver.run()

def get_mapper():
    """Returns standard Jordan-Wigner fermion-to-qubit mapper."""
    return JordanWignerMapper()