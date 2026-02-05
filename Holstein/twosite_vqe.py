import pennylane as qml
from pennylane import numpy as np
from scipy.sparse.linalg import eigsh

t = 1.0 # this is the hopping integral
omega = 0.5 # phonon frequency 
g = 0.5 # electron-phonon coupling strength 

# we are looking at two sites
n_sites = 2
n_wires=2*n_sites  # 2 Electrons + 2 Phonons = 4 Qubits
# Wires 0, 1: Electrons
# Wires 2, 3: Phonons (Local mode 1 on wire 2, Local mode 2 on wire 3)

# We buil the fermionic part of the Hamiltonian 
# Equation: H_el = -t * (c1^d c2 + c2^d c1)
# Only one hopping connection between site 1 and 2
hopping_coeffs = [-t, -t]
hopping_ops = [
    qml.FermiC(0) * qml.FermiA(1), # c1^d c2
    qml.FermiC(1) * qml.FermiA(0)  # c2^d c1
]
# Create Fermi Hamiltonian and Map to Qubits
fermistring = 0
# Create the Fermi Hamiltonian
for i in range(len(hopping_ops)):  
    fermistring = fermistring + hopping_coeffs[i] * hopping_ops[i]  
fermi_ham = fermistring
H_el_qubit = qml.jordan_wigner(fermi_ham)

# Build the Bosonic Part 
# Equation: Sum(b_i^d b_i) for i=1,2
# Mapping (Unary): b^d b -> 0.5 * (I - Z) on phonon wires (2 and 3)
coeffs_ph = []
ops_ph = []
for i in range(n_sites):
    wire_idx = i + n_sites  # Shift: Site 0 -> Wire 2, Site 1 -> Wire 3
    # Term 1: 0.5 * omega * I
    coeffs_ph.append(0.5 * omega)
    ops_ph.append(qml.Identity(wire_idx))
    print(ops_ph)
    # Term 2: -0.5 * omega * Z
    coeffs_ph.append(-0.5 * omega)
    ops_ph.append(qml.PauliZ(wire_idx))
    print(ops_ph)

H_phonon_qubit = qml.Hamiltonian(coeffs_ph, ops_ph)

# We construct the Interaction part: 
# Equation: +g * [ n_1(b_1^d + b_1) + n_2(b_2^d + b_2) ]
# n_i = f_i^d f_i (Electron Density) -> Maps to 0.5*(I - Z)
# (b^d + b) = X (Boson Displacement in Unary basis)

coeffs_int = []
ops_int = []

for i in range(n_sites):
    fermi_wire = i
    boson_wire = i + n_sites

    # Interaction: g * n_i * X_i
    # n_i = 0.5 * (I - Z)
    # Sub-term 1: 0.5 * g * I_el * X_ph
    coeffs_int.append(0.5*g)
    ops_int.append(qml.Identity(fermi_wire) @ qml.PauliX(boson_wire))
    
    # Sub-term 2: -0.5 * g * Z_el * X_ph
    coeffs_int.append(-0.5 * g)
    ops_int.append(qml.PauliZ(fermi_wire) @ qml.PauliX(boson_wire))

H_int_qubit = qml.Hamiltonian(coeffs_int, ops_int)
H_total = H_el_qubit + H_phonon_qubit + H_int_qubit

print("The Full 4-qubit Hamiltonian is:")
print(H_total)

# --- 7. Exact Diagonalization (The "Truth") ---
print("\n--- Running Exact Diagonalization ---")
H_matrix = qml.matrix(H_total)
# k=4 because 4x4 small matrix is trivial, getting lowest ones
# We filter for the 1-electron sector implicitly by looking at the result
eigenvalues, eigenvectors = eigsh(H_matrix, k=4, which='SA')

exact_ground_energy = eigenvalues[0]
print(f"Exact Ground State Energy: {exact_ground_energy:.6f} Ha")

# Run the VQE and look at the ground state energy 
dev = qml.device("default.qubit", wires=n_wires)

def ansatz(params):
    qml.BasisState(np.array([1, 0, 0, 0]), wires=range(n_wires))  # 1. Initialize 1 Electron on Site 1
    #qml.BasisState(np.array([0, 1, 0, 0]), wires=range(n_wires)) # start with 1 electron on site 2
    
    # we apply the variational layers, which will be what will be optimized during the VQE cycle 
    # Purpose: Creating Entanglement and Superposition
    qml.StronglyEntanglingLayers(params, wires=range(n_wires))

@qml.qnode(dev)
def cost_fn(params):
    ansatz(params)
    return qml.expval(H_total)

# This is 
layers = 3 # hyperparameters, s
shape = qml.StronglyEntanglingLayers.shape(n_layers=layers, n_wires=n_wires)
params = np.random.random(shape)

opt = qml.AdamOptimizer(stepsize=0.05)
max_iterations = 500 # if max_iterations is less than that, usually it predicts uneven occupation probabilities

print(f"Starting VQE with {layers} layers...")

for n in range(max_iterations):
    params, energy = opt.step_and_cost(cost_fn, params)
    print(params.shape)
    if n % 20 == 0:
        print(f"Step {n}: Energy = {energy:.6f} Ha")

print(f"\nFinal VQE Energy: {energy:.6f} Ha")

import matplotlib.pyplot as plt
qml.drawer.use_style("black_white")
circuit = qml.QNode(ansatz, dev)
fig, ax = qml.draw_mpl(circuit)(params)
plt.savefig("QC_2site.png")
# Use it as a test to see if it works
@qml.qnode(dev)
def get_probs(params):
    ansatz(params)
    return qml.probs(wires=[0, 1]) # Measure electron position only

final_probs = get_probs(params)
print("\nElectron Probability Distribution:")
print(f"Site 1 (|10>): {final_probs[2]:.4f}") # Binary 10 is index 2
print(f"Site 2 (|01>): {final_probs[1]:.4f}") # Binary 01 is index 1