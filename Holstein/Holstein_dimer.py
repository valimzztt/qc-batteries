import pennylane as qml
from pennylane import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as sp

# We define the Bosonic Operator, which is given by the Number of Phonons
def boson_operators(Nph: int):
    """Return a, adag, n in truncated Fock basis of size Nph."""
    # a|n> = sqrt(n) |n-1>
    data = []
    rows = []
    cols = []
    for n in range(1, Nph):
        rows.append(n - 1)
        cols.append(n)
        data.append(np.sqrt(n))
    a = sp.csr_matrix((data, (rows, cols)), shape=(Nph, Nph), dtype=np.complex128)
    adag = a.conjugate().T
    n_op = adag @ a
    return a, adag, n_op

# ----------------------------
# 2) Pauli matrices as sparse
# ----------------------------
def pauli_sparse():
    I = sp.identity(2, format="csr", dtype=np.complex128)
    X = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.complex128))
    Z = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=np.complex128))
    return I, X, Z

# ----------------------------
# 3) Build Holstein Hamiltonian matrix
# ----------------------------
def holstein_dimer_matrix(t, Delta, omega, g, Nph):
    """
    Fully quantized Holstein dimer (spinless) with one phonon mode truncated to Nph.
    Electron: 2-level system (site 1 vs site 2) -> 2x2
    Phonon: Nph Fock states -> Nph x Nph
    Total dimension: 2*Nph
    """
    I2, X, Z = pauli_sparse()
    a, adag, n_op = boson_operators(Nph)

    Iph = sp.identity(Nph, format="csr", dtype=np.complex128)

    # Terms
    H_hop   = t     * sp.kron(X,  Iph, format="csr")
    print(H_hop)
    H_asym  = Delta * sp.kron(Z,  Iph, format="csr")
    H_ph    = omega * sp.kron(I2, n_op, format="csr")
    H_coup  = g     * sp.kron(Z,  (a + adag), format="csr")

    return (H_hop + H_asym + H_ph + H_coup).tocsr()

# We encode the number of phonons into qubits, using a binary basis
def n_qubits_for_phonon(Nph: int) -> int:
    return int(np.ceil(np.log2(Nph)))

def projector_to_truncated_subspace(Nph: int):
    """
    If Nph is not a power of 2, the phonon register (2^n states) includes extra states.
    This returns a diagonal projector P that keeps only the first Nph basis states
    of the phonon register.
    """
    n = n_qubits_for_phonon(Nph)
    dim_full = 2**n
    diag = np.zeros(dim_full)
    diag[:Nph] = 1.0
    return sp.diags(diag, format="csr", dtype=np.complex128)

def diagonalize_holstein(t, Delta, omega, g, Nph, k_eigs=6):
    H = holstein_dimer_matrix(t, Delta, omega, g, Nph)

    # Compute lowest k eigenvalues/eigenvectors (smallest algebraic)
    # 'SA' = smallest algebraic. For hermitian matrices eigsh is appropriate.
    evals, evecs = spla.eigsh(H, k=k_eigs, which="SA")

    # Sort just to be safe
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]
    return evals, evecs, H

def embed_into_qubits(H_small, Nph):
    """
    Embed 2*Nph Hamiltonian into full qubit Hilbert space of size 2*(2^n).
    This pads the phonon space to the next power of 2 and projects.
    """
    n = n_qubits_for_phonon(Nph)
    dim_ph_full = 2**n

    # Electron space is exactly 2; phonon padded to 2^n
    # Build block embedding: H_embed = (I_e ⊗ P) (H_pad) (I_e ⊗ P)
    # We pad H_small into the top-left (2*Nph) block of (2*2^n).
    dim_small = 2 * Nph
    dim_full  = 2 * dim_ph_full

    # Pad into full space
    H_pad = sp.csr_matrix((dim_full, dim_full), dtype=np.complex128)
    H_pad[:dim_small, :dim_small] = H_small

    P = projector_to_truncated_subspace(Nph)
    P_full = sp.kron(sp.identity(2, format="csr", dtype=np.complex128), P, format="csr")

    return (P_full @ H_pad @ P_full).tocsr()

# We implement the VQE in Pennylane 
t = 1.0 # this is the hopping integral
omega = 0.5 # phonon frequency 
g_true = 0.5 # electron-phonon coupling strength 
g = g_true  * np.sqrt(2)
Delta= 0.00   # eV
Nph = 16   # phonon truncation (always power of 2!)

# Build Hamiltonian
H_small = holstein_dimer_matrix(t, Delta, omega, g, Nph)

print("This is the small Hamiltonian")
print(H_small)



H_full  = embed_into_qubits(H_small, Nph)

n_ph_qubits = n_qubits_for_phonon(Nph)
n_wires = 1 + n_ph_qubits  # 1 electron qubit + phonon register

dev = qml.device("default.qubit", wires=n_wires)


H_pl = qml.SparseHamiltonian(H_full, wires=range(n_wires))

def ansatz(params):
    # electron qubit
    qml.RY(params[0], wires=0)
    qml.RZ(params[1], wires=0)

    # phonon register (hardware-efficient, simple starter)
    idx = 2
    for w in range(1, n_wires):
        qml.RY(params[idx], wires=w); idx += 1
        qml.RZ(params[idx], wires=w); idx += 1

    # entangle electron with phonons
    for w in range(1, n_wires):
        qml.CNOT(wires=[0, w])

@qml.qnode(dev)
def energy(params):
    ansatz(params)
    return qml.expval(H_pl)

# VQE loop
np.random.seed(0)
n_params = 2 + 2*n_ph_qubits
params = np.random.normal(scale=0.1, size=(n_params,), requires_grad=True)

opt = qml.AdamOptimizer(stepsize=0.05)
for it in range(300):
    params = opt.step(energy, params)
    
E0 = energy(params)
print(f"VQE ground energy (Nph={Nph}): {E0:.6f} eV")

gamma = 2*g**2/omega
E_a = gamma/4
print("Expected activation energy:", E_a)

circuit = qml.QNode(ansatz, dev)
import matplotlib.pyplot as plt
qml.drawer.use_style("black_white")
fig, ax = qml.draw_mpl(circuit)(params)
plt.savefig("Quantum_circuit.png")

evals, evecs, H = diagonalize_holstein(t, Delta, omega, g, Nph, k_eigs=6)
print("Lowest eigenvalues (eV):")
for i, E in enumerate(evals):
    print(f"{i}: {E:.6f}")