import pennylane as qml
from pennylane import numpy as np
from scipy.sparse.linalg import eigsh

# --- 1. Physics Parameters ---
t = 1.0       # Hopping
omega = 0.5   # Phonon Frequency
g = 0.5       # Coupling
# We encode the number of phonons into qubits
def n_qubits_for_phonon(Nph: int) -> int:
    return int(np.ceil(np.log2(Nph)))
# --- 2. System Dimensions ---
n_sites = 2                    # 2 Electronic Sites
n_phonons_per_site = 4         # Truncation: 0, 1, 2, 3 phonons
n_qubits_per_phonon = n_qubits_for_phonon(n_phonons_per_site)       # We need 2 qubits to store numbers 0-3 (2^2=4)



# Total Qubit Map
# Wires 0, 1: Electrons
# Wires 2, 3: Phonon Mode 1 (Binary)
# Wires 4, 5: Phonon Mode 2 (Binary)
n_wires = n_sites + (n_sites * n_qubits_per_phonon)

print(f"System: {n_sites} Sites, {n_phonons_per_site} Fock States per Mode")
print(f"Total Qubits: {n_wires}")

# --- 3. Helper: Binary Encoding of Boson Operators ---
def get_boson_paulis(truncation, num_qubits, wires):
    """
    Generates the Pauli-decomposed operators for 'n' (Number) and 'x' (Displacement)
    in the Binary representation on specific wires.
    """
    # A. Build the Matrices in Fock Basis (Size N x N)
    # Number Operator: diag(0, 1, 2, 3)
    mat_n = np.diag(np.arange(truncation))
    
    # Annihilation Operator: a|n> = sqrt(n)|n-1>
    mat_a = np.zeros((truncation, truncation))
    for k in range(1, truncation):
        mat_a[k-1, k] = np.sqrt(k)
        
    # Displacement: x = a^d + a
    mat_x = mat_a.T + mat_a
    
    # B. Pad to match Qubit Dimension (if truncation < 2^num_qubits)
    # (Here 4 fits exactly into 2^2, so no padding needed, but good practice)
    dim_hilbert = 2**num_qubits
    if truncation < dim_hilbert:
        pad_n = np.zeros((dim_hilbert, dim_hilbert))
        pad_x = np.zeros((dim_hilbert, dim_hilbert))
        pad_n[:truncation, :truncation] = mat_n
        pad_x[:truncation, :truncation] = mat_x
    else:
        pad_n = mat_n
        pad_x = mat_x

    # C. Convert Matrix -> Pauli Strings (The Magic Step)
    # This automatically finds the combination of I, X, Y, Z to represent the matrix
    op_n = qml.pauli_decompose(pad_n, wire_order=wires)
    op_x = qml.pauli_decompose(pad_x, wire_order=wires)
    
    return op_n, op_x

# --- 4. Construct Hamiltonian ---

# A. Electronic Hopping (Same as before)
hopping_coeffs = [-t, -t]
hopping_ops = [
    qml.FermiC(0) * qml.FermiA(1), 
    qml.FermiC(1) * qml.FermiA(0)
]
# Create Fermi Hamiltonian and Map to Qubits
fermistring = 0
# Create the Fermi Hamiltonian
for i in range(len(hopping_ops)):  
    fermistring = fermistring + hopping_coeffs[i] * hopping_ops[i]  
fermi_ham = fermistring
H_el_qubit = qml.jordan_wigner(fermi_ham)

# B. Phonon Energy (H_ph = omega * sum(n_i))
# We generate the 'n' operator for each mode using our helper
ops_ph_list = []
coeffs_ph_list = []

for i in range(n_sites):
    # Calculate which wires belong to this phonon mode
    # Mode 0 -> Wires [2, 3]
    # Mode 1 -> Wires [4, 5]
    start = n_sites + (i * n_qubits_per_phonon)
    ph_wires = list(range(start, start + n_qubits_per_phonon))
    
    # Get Binary Operators
    op_n, _ = get_boson_paulis(n_phonons_per_site, n_qubits_per_phonon, ph_wires)
    
    # Add omega * n
    # Note: op_n is already a Hamiltonian/Operator object
    ops_ph_list.append(op_n)
    coeffs_ph_list.append(omega)

# Sum them up
H_phonon_qubit = qml.Hamiltonian(coeffs_ph_list, ops_ph_list)
# C. Interaction (H_int = g * sum( n_el_i * x_ph_i ))
ops_int_list = []
coeffs_int_list = []

for i in range(n_sites):
    el_wire = i
    
    # Phonon wires for this site
    start = n_sites + (i * n_qubits_per_phonon)
    ph_wires = list(range(start, start + n_qubits_per_phonon))
    
    # Get Binary Displacement Operator 'x'
    _, op_x = get_boson_paulis(n_phonons_per_site, n_qubits_per_phonon, ph_wires)
    
    # Electron Density n_el = 0.5 * (I - Z)
    # We construct the interaction product: n_el @ x_ph
    
    # Term 1: 0.5 * g * (I @ x_ph)
    coeffs_int_list.append(0.5 * g)
    ops_int_list.append(qml.Identity(el_wire) @ op_x)
    
    # Term 2: -0.5 * g * (Z @ x_ph)
    coeffs_int_list.append(-0.5 * g)
    ops_int_list.append(qml.PauliZ(el_wire) @ op_x)

H_int_qubit = qml.Hamiltonian(coeffs_int_list, ops_int_list)

# Total Hamiltonian
H_total = H_el_qubit + H_phonon_qubit + H_int_qubit

# --- 5. Exact Diagonalization ---
print("\n--- Running Exact Diagonalization ---")
H_matrix = qml.matrix(H_total)
# Get lowest 4 eigenvalues
eigenvalues, _ = eigsh(H_matrix, k=4, which='SA')
print(f"Exact Ground State Energy: {eigenvalues[0]:.6f} Ha")


# --- 6. VQE Setup ---
print("\n--- Running VQE Optimization ---")
dev = qml.device("default.qubit", wires=n_wires)


def ansatz(params):
    # 1. Initialize: 1 Electron on Site 1, 0 Phonons
    # Electron Wires [0,1]: |10>
    # Phonon Wires [2,3,4,5]: |0000>
    state_init = np.zeros(n_wires, dtype=int)
    state_init[0] = 1 # Electron on Site 1
    qml.BasisState(state_init, wires=range(n_wires))
    
    # 2. Ansatz
    qml.StronglyEntanglingLayers(params, wires=range(n_wires))

@qml.qnode(dev)
def cost_fn(params):
    ansatz(params)
    return qml.expval(H_total)

# Loop
layers = 3 # if I set it to 4, I get a totally wrong occupation 
shape = qml.StronglyEntanglingLayers.shape(n_layers=layers, n_wires=n_wires)
params = np.random.random(shape)

opt = qml.AdamOptimizer(stepsize=0.04)
max_iterations = 1000

print(f"Starting VQE on {n_wires} Qubits...")

for n in range(max_iterations):
    params, energy = opt.step_and_cost(cost_fn, params)
    if n % 20 == 0:
        print(f"Step {n}: Energy = {energy:.6f} Ha")

print(f"Final VQE Energy: {energy:.6f} Ha")

# Use it as a test to see if it works
@qml.qnode(dev)
def get_probs(params):
    ansatz(params)
    return qml.probs(wires=[0, 1]) # Measure electron position only

final_probs = get_probs(params)
print("\nElectron Probability Distribution:")
print(f"Site 1 (|10>): {final_probs[2]:.4f}") # Binary 10 is index 2
print(f"Site 2 (|01>): {final_probs[1]:.4f}") # Binary 01 is index 1