import pennylane as qml
from pennylane import qchem
import os 
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=28"
from jax import numpy as jnp
import jax

#jax.config.update("jax_enable_x64", True)

# 1. Define TiO2 Dimer Geometry (Ti-O-Ti bridge)
symbols = ['Ti', 'O', 'Ti']
# Ti-O distance is typically ~1.9 - 2.0 Angstroms
geometry = jnp.array([[0.0, 0.0, -2.0],
                      [0.0, 0.0,  0.0],
                      [0.0, 0.0,  2.0]])

# We Build the molecule
mol = qchem.Molecule(symbols, geometry, mult=1, load_data=True)

# 3. Active Space Selection
# Titanium (STO-3G) has many orbitals. To keep qubits at ~12:
# We select 1 active electron (the polaron) and 6 active orbitals 
# (focused on the Ti 3d and O 2p hybridization path)
active_electrons = 2
active_orbitals = 6
print("Starting the Hamiltonian")
# 4. Build Hamiltonian
""" H_pauli, qubits = qchem.molecular_hamiltonian(
    symbols,
    geometry,
    # charge=7, 
    mult=1,
    basis="sto-3g",
    active_electrons=active_electrons,
    load_data = True,
    active_orbitals=active_orbitals
)

import pickle
with open("TiO2_Hamiltonian.pkl", "wb") as f:
    pickle.dump(H_pauli, f)

print(f"TiO2 Dimer Qubits: {qubits}")
"""

import pickle
# Load from the file
with open("CO2_Hamiltonian.pkl", "rb") as f:
    H_loaded = pickle.load(f)
qubits = 12
H_pauli = H_loaded # Already mapped to qubit hamiltonian

# We now build the quantum circuit with  the UCCSD ansatz: which is constructed with a se of single and double 
# excitation operators. In Pennylane, SingleExcitation and DoubleExcitation operators are efficient but only
# compatible with the Jordan-Wigner mapping. 
# We need the initial state that has the correct number of electrons. 
# We use the Hartree-Fock state which can be obtained in a user-defined basis 
# For that, we need to specify the number of electrons, the number of orbitals and the desired mapping.
hf_state = qchem.hf_state(active_electrons, qubits, basis="bravyi_kitaev") 
hf_state = qchem.hf_state(active_electrons, qubits, basis="occupation_number") 

# Construct the excitation operator mapping manuallz
from pennylane.fermi import from_string
singles, doubles = qchem.excitations(active_electrons, qubits)

singles_fermi = []
for ex in singles:
    singles_fermi.append(from_string(f"{ex[1]}+ {ex[0]}-")
                       - from_string(f"{ex[0]}+ {ex[1]}-"))

doubles_fermi = []
for ex in doubles:
    doubles_fermi.append(from_string(f"{ex[3]}+ {ex[2]}+ {ex[1]}- {ex[0]}-")
                       - from_string(f"{ex[0]}+ {ex[1]}+ {ex[2]}- {ex[3]}-"))
    
singles_pauli = []
for op in singles_fermi:
    singles_pauli.append(qml.bravyi_kitaev(op, qubits, ps=True))

doubles_pauli = []
for op in doubles_fermi:
    doubles_pauli.append(qml.bravyi_kitaev(op, qubits, ps=True))

    params = jnp.array([0.22347661, 0.0, 0.0])

dev = qml.device("default.qubit", wires=qubits)

@qml.qnode(dev)
def circuit(params):
    qml.BasisState(hf_state, wires=range(qubits))

    for i, excitation in enumerate(doubles_pauli):
        qml.exp((excitation * params[i] / 2).operation()), range(qubits)

    for j, excitation in enumerate(singles_pauli):
        qml.exp((excitation * params[i + j + 1] / 2).operation()), range(qubits)

    return qml.expval(H_pauli)
print('Energy =', circuit(params)) 