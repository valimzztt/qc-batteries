import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Model builder (your 3-site + 3 two-level phonons)
# -----------------------------
def build_H_total(t, omega, g):
    # Fermionic hopping on sites 0,1,2 (spinless)
    hopping_ops = [
        qml.FermiC(0) * qml.FermiA(1),
        qml.FermiC(1) * qml.FermiA(0),
        qml.FermiC(1) * qml.FermiA(2),
        qml.FermiC(2) * qml.FermiA(1),
        qml.FermiC(2) * qml.FermiA(0),
        qml.FermiC(0) * qml.FermiA(2),
    ]
    hopping_coeffs = [-t] * len(hopping_ops)

    fermistring = 0
    for c, op in zip(hopping_coeffs, hopping_ops):
        fermistring = fermistring + c * op

    # Map fermions -> qubits (acts on wires 0,1,2)
    H_el_qubit = qml.jordan_wigner(fermistring)

    # 3 local two-level phonons on wires 3,4,5
    # number operator in 0/1 truncation: n = (I - Z)/2
    coeffs_ph, ops_ph = [], []
    for i in range(3):
        w = i + 3
        coeffs_ph.append(0.5 * omega);  ops_ph.append(qml.Identity(w))
        coeffs_ph.append(-0.5 * omega); ops_ph.append(qml.PauliZ(w))
    H_ph = qml.Hamiltonian(coeffs_ph, ops_ph)

    # Holstein coupling: g * n_i * (b_i + b_i^†)
    # In 0/1 truncation: n_i -> (I - Z_i)/2  and  (b+b†) -> X_(i+3)
    coeffs_int, ops_int = [], []
    for i in range(3):
        f = i       # fermion wire i
        b = i + 3   # phonon wire i+3

        # + (g/2) * I_f * X_b
        coeffs_int.append(0.5 * g)
        ops_int.append(qml.Identity(f) @ qml.PauliX(b))

        # - (g/2) * Z_f * X_b
        coeffs_int.append(-0.5 * g)
        ops_int.append(qml.PauliZ(f) @ qml.PauliX(b))

    H_int = qml.Hamiltonian(coeffs_int, ops_int)

    return H_el_qubit + H_ph + H_int


# -----------------------------
# Diagonalization utility
# -----------------------------
def lowest_eigs_vs_g(t, omega, g_values, k=8):
    """
    Returns an array E[g_index, eig_index] of the lowest k eigenvalues.
    Total wires = 6 (3 fermion qubits + 3 phonon qubits).
    """
    wires = list(range(6))
    E = []

    for g in g_values:
        H = build_H_total(t, omega, g)

        # Build dense matrix (64x64); fine for this size
        Hmat = qml.matrix(H, wire_order=wires)

        # Hermitian eigen-decomposition
        evals = np.linalg.eigvalsh(Hmat)
        evals = np.sort(np.real(evals))

        E.append(evals[:k])

    return np.array(E)


# -----------------------------
# Run + plot
# -----------------------------
t = 1.0
omega = 0.5

g_values = np.linspace(0.0, 3.5, 15)   # adjust range/density as you like
k_show = 6                              # number of eigenvalues to plot

E = lowest_eigs_vs_g(t, omega, g_values, k=k_show)
print("Lowest eigenstate", E)
