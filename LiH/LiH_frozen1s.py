# This freezes the lowest energy orbital (Li 1s). We only simulate the remaining 2 valence electrons in the remaining 5 orbitals (10 qubits).
from pennylane import qchem
from jax import numpy as jnp
import jax
import pennylane as qml

jax.config.update("jax_enable_x64", True)

# We define the geometry for LiH, bond length is 3 Bohr
symbols = ['Li', 'H']
geometry = jnp.array([[0.0, 0.0, 0.0],
                      [0.0, 0.0, 3.0]])

# We build the molecule 
mol = qchem.Molecule(symbols, geometry)

# Active space is given by 4 electrons
# Core Electrons (Li 1s): 2 # Active Electrons: 2
active_electrons = 2
# Total Orbitals (STO-3G): 6 Core Orbitals (Li 1s): 1, Active Orbitals: 6 - 1 = 5
active_orbitals = 5

# --- Build Frozen Core Hamiltonian ---
H_frozen, qubits_frozen = qchem.molecular_hamiltonian(
    symbols,
    geometry,
    # active_electrons=active_electrons,
    active_orbitals=active_orbitals
)

print(f"Frozen Core Hamiltonian Qubits: {qubits_frozen}")
# Expected: 10 qubits (5 orbitals * 2 spin states)
core_indices = [0]
active_indices = [1,2,5]
h_fermi =  qchem.fermionic_hamiltonian(mol, core=core_indices, active=active_indices)()
qubits = len(h_fermi.wires)
h_pauli = qml.bravyi_kitaev(h_fermi, qubits, tol=1e-16)
h_pauli = qml.jordan_wigner(h_fermi )
print(h_pauli)

# We need the initial state that has the correct number of electrons. 
# We use the Hartree-Fock state which can be obtained in a user-defined basis 
# For that, we need to specify the number of electrons, the number of orbitals and the desired mapping.
hf_state = qchem.hf_state(active_electrons, qubits, basis="bravyi_kitaev") 
hf_state = qchem.hf_state(active_electrons, qubits, basis="occupation_number") 
# We now build the quantum circuit with  the UCCSD ansatz: which is constructed with a se of single and double 
# excitation operators. In Pennylane, SingleExcitation and DoubleExcitation operators are efficient but only
# compatible with the Jordan-Wigner mapping. 

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

    return qml.expval(h_pauli)

print('Energy =', circuit(params))

