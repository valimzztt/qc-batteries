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
symbols = ['Ti', 'O', 'Ti']
geometry = jnp.array([[3.2, 3.2, 4.439],
                      [2.3, 2.3, 2.959],
                      [3.2, 3.2, 1.480]])

# Assume that your Ti-O-Ti molecule is on the 
# Charge=0, Multiplicity=1 (Singlet) are defaults
molecule = qml.qchem.Molecule(symbols, geometry, load_data=True) # creates 12 spin-orbitals, 43 basis functions
# molecule = qml.qchem.Molecule(symbols, geometry, basis_name="6-31G", load_data=True) # creates 22 spin-orbitals. 67 basis functions
electrons = 52 # Total electrons in TiO2 molecule
orbitals = 35
core_indices, active_indices= qchem.active_space(electrons, orbitals, active_electrons=2, active_orbitals=6)
print("Core indices", core_indices)
print("Active indices", active_indices)
h_fermi =  qchem.fermionic_hamiltonian(molecule, core=core_indices, active= active_indices)()
print("The fermionic hamiltonian is:", h_fermi)

with open("Fermionic_TiO2.pkl", "wb") as f:
    pickle.dump(h_fermi, f)
    
h_pauli = qml.jordan_wigner(h_fermi)
print(h_pauli)


with open("Pauli_TiO2_JW.pkl", "wb") as f:
    pickle.dump(h_pauli, f)