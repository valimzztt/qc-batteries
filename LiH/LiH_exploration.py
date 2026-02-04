from jax import numpy as np
import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update('jax_enable_x64', True)

import pennylane as qml
from pennylane import qchem
from jax import numpy as jnp

symbols = ['Li', 'H']
geometry = jnp.array([[0.0, 0.0, 0.0],
                      [0.0, 0.0, 3.0]])

# Charge=0, Multiplicity=1 (Singlet) are defaults
mol = qchem.Molecule(symbols, geometry)
molecule = qml.qchem.Molecule(symbols, geometry) # creates 12 spin-orbitals
# molecule = qml.qchem.Molecule(symbols, geometry, basis_name="6-31G") creates 22 spin-orbitals
print(molecule.mult)
h_fermi =  qchem.fermionic_hamiltonian(molecule)()
print("The fermionic hamiltonian is:", h_fermi)
h_pauli = qml.jordan_wigner(h_fermi)
# print("The Pauli Hamiltonian is:", h_pauli)


electrons = 4
orbitals = 6
core_indices, active_indices= qchem.active_space(electrons, orbitals, active_electrons=2, active_orbitals=5)
print(core_indices)
print("Active indices", active_indices)



