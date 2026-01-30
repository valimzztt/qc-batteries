import pennylane as qml
from pennylane import qchem
from jax import numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)

# 1. Define TiO2 Dimer Geometry (Ti-O-Ti bridge)
symbols = ['O', 'C', 'O']
# CO2 Bond length ~1.16 Angstroms
geometry = jnp.array([[0.0, 0.0, -1.16],
                      [0.0, 0.0,  0.0],
                      [0.0, 0.0,  1.16]])

# We Build the molecule
mol = qchem.Molecule(symbols, geometry, mult=1, load_data=True)

# 3. Active Space Selection
# Titanium (STO-3G) has many orbitals. To keep qubits at ~12:
# We select 1 active electron (the polaron) and 6 active orbitals 
# (focused on the Ti 3d and O 2p hybridization path)
active_electrons = 2
active_orbitals = 5

# 4. Build Hamiltonian
H_pauli, qubits = qchem.molecular_hamiltonian(
    symbols,
    geometry,
    mult=1,
    basis="sto-3g",
    active_electrons=active_electrons,
    load_data = True,
    active_orbitals=active_orbitals
)

print(f"CO2 Dimer Qubits: {qubits}")
# Mapping to Pauli operators for VQE
print(H_pauli)

import pickle

# After building your Hamiltonian:
# H, qubits = qml.qchem.molecular_hamiltonian(...)

# Save to a file
with open("CO2_Hamiltonian.pkl", "wb") as f:
    pickle.dump(H_pauli, f)

print("Hamiltonian saved successfully!")

# We now build the quantum circuit with  the UCCSD ansatz: which is constructed with a se of single and double 
# excitation operators. In Pennylane, SingleExcitation and DoubleExcitation operators are efficient but only
# compatible with the Jordan-Wigner mapping. 
# We need the initial state that has the correct number of electrons. 
# We use the Hartree-Fock state which can be obtained in a user-defined basis 
# For that, we need to specify the number of electrons, the number of orbitals and the desired mapping.
hf_state = qchem.hf_state(active_electrons, qubits, basis="bravyi_kitaev") 
hf_state = qchem.hf_state(active_electrons, qubits, basis="occupation_number") 
# Construct the excitation operator mapping manuallz

singles, doubles = qchem.excitations(electrons, qubits)

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

    return qml.expval(h_pauli)

print('Energy =', circuit(params))