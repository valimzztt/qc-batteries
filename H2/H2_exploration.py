from jax import numpy as np
import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update('jax_enable_x64', True)
import pennylane as qml
from pennylane import qchem


symbols = ["H", "H"]
coordinates = np.array([[-0.70108983, 0.0, 0.0], [0.70108983, 0.0, 0.0]])
molecule = qml.qchem.Molecule(symbols, coordinates)
molecule = qml.qchem.Molecule(symbols, coordinates, basis_name="6-31G")
print(molecule.mult)
h_fermi =  qchem.fermionic_hamiltonian(molecule)()
print("The fermionic hamiltonian is:", h_fermi)
h_pauli = qml.jordan_wigner(h_fermi)
print(h_pauli)