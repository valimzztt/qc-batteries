import pennylane as qml
from pennylane import qchem
from pennylane.fermi import from_string
import jax
from jax import numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)
symbols = ['Ti', 'O', 'Ti']
geometry = jnp.array([[3.2, 3.2, 4.439],
                      [2.3, 2.3, 2.959],
                      [3.2, 3.2, 1.480]])

# Assume that your Ti-O-Ti molecule is on the 
# Charge=0, Multiplicity=1 (Singlet) are defaults
molecule = qml.qchem.Molecule(symbols, geometry, load_data=True) # creates 12 spin-orbitals
# molecule = qml.qchem.Molecule(symbols, geometry, basis_name="6-31G") creates 22 spin-orbitals
print(molecule.mult)
electrons = 52 # Total electrons in TiO2 molecule
orbitals = 35
core_indices, active_indices= qchem.active_space(electrons, orbitals, active_electrons=2, active_orbitals=6)

h_fermi =  qchem.fermionic_hamiltonian(molecule, core=core_indices, active= active_indices)()
print("The fermionic hamiltonian is:", h_fermi)
h_pauli = qml.jordan_wigner(h_fermi)
print(h_pauli)