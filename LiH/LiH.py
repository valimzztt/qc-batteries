from pennylane import qchem
from jax import numpy as jnp
import jax
import pennylane as qml
from pennylane.fermi import from_string

jax.config.update("jax_enable_x64", True)
# We start by defining the geometry of the LiH molecule
# Bond length is approx. 3 Bohr
symbols = ['Li', 'H']
geometry = jnp.array([[0.0, 0.0, 0.0],
                      [0.0, 0.0, 3.0]])

# Charge=0, Multiplicity=1 (Singlet) are defaults
mol = qchem.Molecule(symbols, geometry)
# Let´s start by first building the fermionic Hamiltonian: 
# 1. Get Fermionic Hamiltonian (e.g. from integrals)
h_fermi = qchem.fermionic_hamiltonian(mol)()
electrons = 3
qubits = len(h_fermi.wires)
# Transform the fermionic Hamiltonian to its qubit representation via Bravji-Kitaev transformation
h_pauli = qml.bravyi_kitaev(h_fermi, qubits, tol=1e-16)
h_pauli = qml.jordan_wigner(h_fermi )

# We use the Hartree-Fock state which can be obtained in a user-defined basis 
# For that, we need to specify the number of electrons, the number of orbitals and the desired mapping.
hf_state = qchem.hf_state(electrons, qubits, basis="bravyi_kitaev") 
hf_state = qchem.hf_state(electrons, qubits, basis="occupation_number") 

# We now build the quantum circuit with  the UCCSD ansatz: which is constructed with a se of single and double 
# excitation operators. In Pennylane, SingleExcitation and DoubleExcitation operators are efficient but only
# compatible with the Jordan-Wigner mapping. 

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
