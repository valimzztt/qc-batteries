import numpy as np

from qiskit import Aer
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Estimator

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.algorithms import GroundStateEigensolver


# --------------------------------------------------
# 1. Define LiH molecule (geometry in Angstrom)
# --------------------------------------------------
driver = PySCFDriver(
    atom="Li 0 0 0; H 0 0 1.6",
    basis="sto3g",
    spin=0,
    charge=0
)

problem = driver.run()

# Optional: reduce problem size (very common in QC papers)
transformer = ActiveSpaceTransformer(
    num_electrons=2,
    num_spatial_orbitals=2
)
problem = transformer.transform(problem)

# --------------------------------------------------
# 2. Map fermions → qubits
# --------------------------------------------------
mapper = JordanWignerMapper()
hamiltonian = mapper.map(problem.hamiltonian)

num_qubits = hamiltonian.num_qubits
print(f"Number of qubits: {num_qubits}")

# --------------------------------------------------
# 3. Define VQE ansatz + optimizer
# --------------------------------------------------
ansatz = TwoLocal(
    num_qubits,
    rotation_blocks="ry",
    entanglement_blocks="cz",
    entanglement="linear",
    reps=2
)

optimizer = SLSQP(maxiter=200)

# --------------------------------------------------
# 4. Run VQE
# --------------------------------------------------
estimator = Estimator()
vqe = VQE(
    estimator=estimator,
    ansatz=ansatz,
    optimizer=optimizer
)

solver = GroundStateEigensolver(mapper, vqe)
result = solver.solve(problem)

# --------------------------------------------------
# 5. Output energy
# --------------------------------------------------
print("Ground-state energy (Hartree):")
print(result.total_energies[0])
