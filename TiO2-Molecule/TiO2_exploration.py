import pennylane as qml
from pennylane import qchem
from pennylane.fermi import from_string
import os 
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
import jax
from jax import numpy as jnp
import numpy as np
import pickle


jax.config.update('jax_enable_x64', True)
symbols = ['Ti', 'O', 'O']

geometry = jnp.array([
    [0.000, 0.000, 0.000],   # Ti
    [1.620, 0.000, 0.000],   # O
    [-0.810, 0.000, 1.403],  # O (≈112° angle)
])

# Assume that your Ti-O-Ti molecule is on the 
# Charge=0, Multiplicity=1 (Singlet) are defaults
molecule = qml.qchem.Molecule(symbols, geometry, load_data=True) # creates 12 spin-orbitals, 43 basis functions
print("The number of atomic basis functions per atom: ", molecule.n_basis) 
orbitals = sum(molecule.n_basis) # Only valid fpr the simplest basis: total number of orbitals for the
print("The number of atomic basis functions: ", orbitals) 
# molecule = qml.qchem.Molecule(symbols, geometry, basis_name="6-31G", load_data=True) # creates 22 spin-orbitals. 67 basis functions
electrons = 38 # Total electrons in TiO2 molecule
# The number of orbitals depend on how many molecular orbitals come out from the Hartree fock
active_electrons = 4
active_orbitals = 6
core_indices, active_indices= qchem.active_space(electrons, orbitals=orbitals, active_electrons=active_electrons, active_orbitals=active_orbitals)
print("Core indices", core_indices)
print("Active indices", active_indices)
active_orbitals = orbitals
h_pauli, qubits = qchem.molecular_hamiltonian(
    symbols, geometry,
    method = "pyscf", load_data=True
)

core_indices, active_indices= qchem.active_space(electrons, orbitals=orbitals, active_electrons=active_electrons, active_orbitals=active_orbitals)
print("Core indices", core_indices)
print("Active indices", active_indices)

h_fermi =  qchem.fermionic_hamiltonian(molecule, core=core_indices, active= active_indices)()
print("The fermionic hamiltonian is:", h_fermi)

with open("Fermionic_MoleculeTiO2.pkl", "wb") as f:
    pickle.dump(h_fermi, f)
    
h_pauli = qml.jordan_wigner(h_fermi)
print(h_pauli)


with open("Pauli_MoleculeTiO2_JW.pkl", "wb") as f:
    pickle.dump(h_pauli, f)