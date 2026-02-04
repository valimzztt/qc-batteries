import pennylane as qml
from pennylane import numpy as np

t = 1.0     
omega = 0.5 # Phonon frequency
g = 0.2     # Electron-phonon coupling strength

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

print("Qubit Hamiltonian")
print(H_el_qubit)


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

# --- 5. Combine Everything ---
H_total = H_el_qubit + H_phonon_qubit + H_int_qubit

print("\n--- Full Hamiltonian (6 Qubits) ---")
print(H_total)