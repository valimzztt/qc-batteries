
import pennylane as qml
from pennylane import qchem
from jax import numpy as jnp
import jax
import pickle
#jax.config.update('jax_num_cpu_devices', 8)
jax.config.update("jax_enable_x64", True)

# 1. Define TiO2 Dimer Geometry (Ti-O-Ti bridge)
symbols = ['Ti', 'O', 'Ti']
""" These are the coordinates from MP:
(3.2, 3.2, 4.439) 
(2.3, 2.3, 2.959)
(3.2, 3.2, 1.480) """


# Coordinates taken from MP
geometry = jnp.array([[3.2, 3.2, 4.439],
                      [2.3, 2.3, 2.959],
                      [3.2, 3.2, 1.480]])

# We Build the molecule
mol = qchem.Molecule(symbols, geometry, mult=1, load_data=True)

with open("TiO2_Hamiltonian_pyscf.pkl", "rb") as f:
    H_pauli = pickle.load(f)
    
# 3. Active Space Selection
# Titanium (STO-3G) has many orbitals. To keep qubits at ~12:
# We select 1 active electron (the polaron) and 6 active orbitals 
# (focused on the Ti 3d and O 2p hybridization path)
active_electrons = 2
active_orbitals = 6
# We now build the quantum circuit with  the UCCSD ansatz: which is constructed with a se of single and double 
# excitation operators. In Pennylane, SingleExcitation and DoubleExcitation operators are efficient but only
# compatible with the Jordan-Wigner mapping. 
# We need the initial state that has the correct number of electrons. 
# We use the Hartree-Fock state which can be obtained in a user-defined basis 
# For that, we need to specify the number of electrons, the number of orbitals and the desired mapping.
# Expected: 10 qubits (5 orbitals * 2 spin states)
# Define active space
electrons = 52 # Total electrons in TiO2 molecule
orbitals = 35
core_indices, active_indices= qchem.active_space(electrons, orbitals, active_electrons=2, active_orbitals=6)
print("Core orbitals:", core_indices)
print("Active orbitals:", active_indices)


# h_fermi =  qchem.fermionic_hamiltonian(mol, core=core_indices, active=active_indices)()
qubits = len(H_pauli.wires)
# h_pauli = qml.bravyi_kitaev(H_pauli, qubits, tol=1e-16)
# h_pauli = qml.jordan_wigner(h_fermi )
h_pauli = H_pauli

# We need the initial state that has the correct number of electrons. 
# We use the Hartree-Fock state which can be obtained in a user-defined basis 
# For that, we need to specify the number of electrons, the number of orbitals and the desired mapping.
#hf_state = qchem.hf_state(active_electrons, qubits, basis="bravyi_kitaev") 
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

