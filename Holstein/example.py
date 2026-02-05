import pennylane as qml
from pennylane import numpy as np

t = 1.0     
omega = 0.5 # Phonon frequency
g = 0.0    # Electron-phonon coupling strength

# --- 2. Construct the Fermionic Part (H_el) ---
# Equation: H_el = -f1^d f2 - f2^d f1 - f2^d f3 - f3^d f2 - f3^d f1 - f1^d f3
# Wires 0, 1, 2 correspond to sites 1, 2, 3

# Define the Fermi words using PennyLane's native Fermi operators
# 0 -> site 1, 1 -> site 2, 2 -> site 3
hopping_coeffs = [-t] * 6
hopping_ops = [
    qml.FermiC(0) * qml.FermiA(1), # f1^d f2
    qml.FermiC(1) * qml.FermiA(0), # f2^d f1
    qml.FermiC(1) * qml.FermiA(2), # f2^d f3
    qml.FermiC(2) * qml.FermiA(1), # f3^d f2
    qml.FermiC(2) * qml.FermiA(0), # f3^d f1
    qml.FermiC(0) * qml.FermiA(2)  # f1^d f3
]
fermistring = 0
# Create the Fermi Hamiltonian
for i in range(len(hopping_ops)):  
    fermistring = fermistring + hopping_coeffs[i] * hopping_ops[i]  

# H_fermi = qml.ops.LinearCombination(hopping_coeffs, hopping_ops)
# Map to Qubits using Jordan-Wigner
# This matches the specific Pauli strings in your second image (Z terms appear for ordering)
H_el_qubit = qml.jordan_wigner(fermistring)

# --- 3. Construct the Bosonic Part (H_ph) ---
# Equation: Sum(b_i^d b_i)
# Mapping (Unary/One-to-One from image 3): 
# Each site has 1 boson qubit. State |1> means 1 boson, |0> means 0 bosons.
# b^d b -> Number operator -> 0.5 * (I - Z)
# Wires 3, 4, 5 correspond to phonon modes on sites 1, 2, 3

coeffs_ph = []
ops_ph = []

for i in range(3):
    wire_idx = i + 3  # Shift to wires 3,4,5
    
    # Construct Number Operator: n = 0.5 * (I - Z)
    # Term 1: 0.5 * I
    coeffs_ph.append(0.5 * omega)
    ops_ph.append(qml.Identity(wire_idx))
    
    # Term 2: -0.5 * Z
    coeffs_ph.append(-0.5 * omega)
    ops_ph.append(qml.PauliZ(wire_idx))

H_phonon_qubit = qml.Hamiltonian(coeffs_ph, ops_ph)


# --- 4. Construct the Interaction Part (H_int) ---
# Equation: +g * [ n_1(b_1^d + b_1) + n_2(...) + n_3(...) ]
# n_i = f_i^d f_i  (Electron Density)
# (b^d + b) = X    (Boson Displacement in Unary basis)

coeffs_int = []
ops_int = []

for i in range(3):
    fermi_wire = i
    boson_wire = i + 3
    
    # Step A: Create Electron Density Operator (n_i) in Qubit basis
    # n = f^d f maps to 0.5*(I - Z) in Jordan-Wigner
    # We build this manually to mix with the boson operator
    
    # The term is: g * 0.5 * (I - Z)_fermi * X_boson
    
    # Sub-term 1: g * 0.5 * I_fermi * X_boson
    coeffs_int.append(0.5 * g)
    ops_int.append(qml.Identity(fermi_wire) @ qml.PauliX(boson_wire))
    
    # Sub-term 2: -g * 0.5 * Z_fermi * X_boson
    coeffs_int.append(-0.5 * g)
    ops_int.append(qml.PauliZ(fermi_wire) @ qml.PauliX(boson_wire))

H_int_qubit = qml.Hamiltonian(coeffs_int, ops_int)
H_total = H_el_qubit + H_phonon_qubit + H_int_qubit

print("The Full 6 qubits Hamiltonian is")
print(H_total)
n_wires = 6
# We first run an exact diagonalization
H_el = qml.matrix(H_el_qubit)
print(H_el.shape)
print(H_el)
print(H_phonon_qubit)
H_matrix = qml.matrix(H_total)
print(H_matrix.shape)
from scipy.sparse.linalg import eigsh
# 2. Diagonalize it to find the lowest eigenvalues
#    'eigsh' is efficient for finding the k smallest eigenvalues (k=1)
#    which='SA' means "Smallest Algebraic" (most negative)
eigenvalues, eigenvectors = eigsh(H_matrix, k=20, which='SA')
print(eigenvalues)
exact_ground_energy = eigenvalues[0]
exact_ground_state = eigenvectors[:, 0]

print(f"Exact Ground State Energy: {exact_ground_energy:.6f} Ha")
# Let´s now compute the ground state of this Hamiltonian

dev = qml.device("default.qubit", wires=n_wires)

# --- 6. VQE Setup ---
print("\n--- Running VQE Optimization ---")

# Define the device
dev = qml.device("default.qubit", wires=6)

# Define the Ansatz (The Trial Wavefunction)
# We use StrongEntanglingLayers as a general-purpose ansatz
# + BasisState init to ensure we are in the 1-electron sector
def ansatz(params):
    # 1. Initialize 1 Electron on Site 1 (Wires 0,1,2 are Fermions)
    #    State: |1 0 0 | 0 0 0>
    qml.BasisState(np.array([0, 0, 0, 0, 0, 0]), wires=range(6))
    
    # 2. Apply variational layers to explore the Hilbert space
    #    This entangles the electron position with the phonon states
    qml.StronglyEntanglingLayers(params, wires=range(6))

# Define the Cost Function (Expectation Value of H)
@qml.qnode(dev)
def cost_fn(params):
    ansatz(params)
    return qml.expval(H_total)

# --- 7. The Optimization Loop ---
# Initialize random parameters
# 3 layers, 6 wires, 3 params per rotation
layers = 3
shape = qml.StronglyEntanglingLayers.shape(n_layers=layers, n_wires=6)
params = np.random.random(shape)

# Use the Adam Optimizer (standard for VQE)
opt = qml.AdamOptimizer(stepsize=0.05)
max_iterations = 1000

print(f"Starting VQE with {layers} layers...")

for n in range(max_iterations):
    params, energy = opt.step_and_cost(cost_fn, params)
    
    if n % 10 == 0:
        print(f"Step {n}: Energy = {energy:.6f} Ha")

print(f"\nFinal VQE Energy: {energy:.6f} Ha")

# --- 8. Analyze the Result ---
# Let's peek at the final state to see where the electron is
@qml.qnode(dev)
def get_probs(params):
    ansatz(params)
    return qml.probs(wires=[0, 1, 2]) # Measure electron position

final_probs = get_probs(params)
print("\nElectron Probability Distribution [Site 1, Site 2, Site 3]:")
# Note: probs gives probabilities for 000, 001, 010, 011... 
# We care about states with 1 electron: |100> (index 4), |010> (index 2), |001> (index 1)
print(f"Site 1 (|100>): {final_probs[4]:.4f}")
print(f"Site 2 (|010>): {final_probs[2]:.4f}")
print(f"Site 3 (|001>): {final_probs[1]:.4f}")