import pennylane as qml
from pennylane import numpy as np

# We define the Fe-O-Fe dimer geometrtry (values in angstrom)
# Initial state: Oxygen closer to Fe1
symbols = ["Fe", "O", "Fe"]
coordinates = np.array([
    [0.0, 0.0, -1.5], # Fe 1
    [0.0, 0.0, -0.6], # Bridging O (asymmetric)
    [0.0, 0.0,  1.5]  # Fe 2
])

basis = "sto-3g" # '6-31g'

# 2. Setup the Molecular Hamiltonian with an Active Space
# We focus on the valence electrons in the d and p orbitals
H, qubits = qml.qchem.molecular_hamiltonian(
    symbols, 
    coordinates, 
    charge=2,             # Charge for Fe2+/Fe3+ balance
    mult=1,               # Doublet state (one unpaired electron)
    basis="sto-3g" ,       # Minimal basis to keep qubit count low
    load_data = True,
    active_electrons=2,   # The single hopping electron/hole
    active_orbitals=6,    # 5 Iron d-orbitals + 1 Oxygen p-orbital

)

print(f"Number of qubits required: {qubits}")