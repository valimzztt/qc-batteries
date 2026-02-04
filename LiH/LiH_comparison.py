# This code compares how the ground state energy for LiH changes as a function of the orbitals 
# included in the calculation

import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np
import jax
import jax.numpy as jnp
import time
from pennylane.fermi import from_string

# Enable JAX 64-bit mode for precision
jax.config.update("jax_enable_x64", True)

# --- 1. Define Molecule Geometry ---
symbols = ['Li', 'H']
# Bond length 3.0 Bohr
geometry = jnp.array([[0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.64]])

# --- 2. Helper Function for VQE ---
def solve_lih_config(name,active_electrons, core_indices, active_indices):
    print(f"\n--- Running Configuration: {name} ---")
    print(f"Index of active Electrons: {core_indices}")
    print(f"Index of active Orbitals: {active_indices}")

    # 1. Build Hamiltonian
    # qchem.molecular_hamiltonian automatically handles the freezing of orbitals 
    # that are not selected in the 'active_orbitals' count (assuming canonical ordering).
    # We build the molecule 
    mol = qchem.Molecule(symbols, geometry)
    h_fermi =  qchem.fermionic_hamiltonian(mol, core=core_indices, active=active_indices)()
    qubits = len(h_fermi.wires)
    # Transform the fermionic Hamiltonian to its qubit representation via Bravji-Kitaev transformation
    h_pauli = qml.bravyi_kitaev(h_fermi, qubits, tol=1e-16)
    h_pauli = qml.jordan_wigner(h_fermi)
    
    print(f"Qubits required: {qubits}")
    # 2. Define Hartree-Fock State
    # This state prepares the 'reference' configuration of electrons in the active space
    hf_state = qchem.hf_state(active_electrons, qubits, basis="occupation_number") 
    # 3. Generate Excitations (Singles and Doubles)
    singles, doubles = qchem.excitations(active_electrons, qubits)
    print(f" Ansatz Terms: {len(singles)} singles, {len(doubles)} doubles")

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
    energy = circuit(params)
    return energy

# Case 1: All Orbitals
# LiH STO-3G has 6 spatial orbitals. 4 Electrons total.
e1 = solve_lih_config(
    "1) All Orbitals (Full)", 
    active_electrons=4,
    core_indices = [], 
    active_indices = [0, 1,2,3, 4, 5]
)

# Case 2: Frozen Core (Li 1s removed), Active px/py/pz/s
# We remove 1 orbital (Core) and 2 electrons (Core).
# Remaining: 6-1= 5 orbitals. 4 - 2 = 2 electrons.
e2 = solve_lih_config(
    "2) Frozen Core (Active px/py included)", 
    active_electrons=2, 
    core_indices = [0],
    active_indices = [1,2,3, 4, 5]
)


# Case 3: Frozen Core AND Inactive px/py
# We want to remove the Core (1 orbital) AND the two pi orbitals (2 orbitals).
# Total active orbitals = 6 - 1 - 2 = 3.
# This leaves only the Sigma bonding/antibonding type orbitals in the active space.
e3 = solve_lih_config(
    "3) Frozen Core + Inactive px/py", 
    active_electrons=2, 
    core_indices = [0],
    active_indices = [1,2,5]
)

#
print("             RESULTS SUMMARY")
print("="*40)
print(f"1) Full Config:        {e1:.15f} Ha")
print(f"2) Frozen Core:        {e2:.15f} Ha")
print(f"3) Sigma Only (No pi): {e3:.15f} Ha")
print("-" * 40)
print(f"Error (Frozen vs Full):  {abs(e2 - e1):.8f} Ha")
print(f"Error (Sigma vs Frozen): {abs(e3 - e2):.8f} Ha")
print("="*40)