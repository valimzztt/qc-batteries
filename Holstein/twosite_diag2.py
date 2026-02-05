import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ---------- Fermion operators for 2 spinless modes ----------
def fermion_ops_two_sites():
    """
    Returns creation/annihilation ops c1,c1†, c2,c2† and number ops n1,n2
    in the full 2-mode Fock space (dimension 4) with Jordan-Wigner signs built in.
    Basis ordering: |n1 n2> with n1,n2 in {0,1} -> |00>,|01>,|10>,|11>
    """
    # Single-qubit matrices
    I = sp.identity(2, format="csr", dtype=np.complex128)
    Z = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=np.complex128))
    sp_plus  = sp.csr_matrix(np.array([[0, 1], [0, 0]], dtype=np.complex128))  # |1><0|
    sp_minus = sp.csr_matrix(np.array([[0, 0], [1, 0]], dtype=np.complex128))  # |0><1|

    # Jordan-Wigner mapping for spinless fermions:
    # c1 = σ^- ⊗ I
    # c2 = Z ⊗ σ^-
    c1  = sp.kron(sp_minus, I, format="csr")
    cd1 = sp.kron(sp_plus,  I, format="csr")

    c2  = sp.kron(Z, sp_minus, format="csr")
    cd2 = sp.kron(Z, sp_plus,  format="csr")

    n1 = cd1 @ c1
    n2 = cd2 @ c2
    I4 = sp.identity(4, format="csr", dtype=np.complex128)

    return c1, cd1, c2, cd2, n1, n2, I4

# ---------- Boson operators (1 mode, Nph levels) ----------
def boson_ops(Nph):
    data, rows, cols = [], [], []
    for n in range(1, Nph):
        rows.append(n-1); cols.append(n); data.append(np.sqrt(n))
    a = sp.csr_matrix((data, (rows, cols)), shape=(Nph, Nph), dtype=np.complex128)
    adag = a.conjugate().T
    n = adag @ a
    I = sp.identity(Nph, format="csr", dtype=np.complex128)
    return a, adag, n, I

# ---------- Build Hamiltonian ----------
def holstein_2site_1phonon_matrix(t, omega, g, Nph):
    c1, cd1, c2, cd2, n1, n2, I4 = fermion_ops_two_sites()
    a, adag, nph, Iph = boson_ops(Nph)

    # Electron hopping term
    H_el = -t * (cd1 @ c2 + cd2 @ c1)   # (c1† c2 + c2† c1)

    # Phonon energy
    H_ph = omega * nph

    # Coupling: (n1 - n2)(a + a†)
    H_int = g * sp.kron((n1 - n2), (a + adag), format="csr")

    # Embed electron-only and phonon-only terms in product space
    H = sp.kron(H_el, Iph, format="csr") + sp.kron(I4, H_ph, format="csr") + H_int
    return H.tocsr()

# ---------- Diagonalize ----------
t = 1.0
omega = 0.5
g = 0.5
Nph = 2

H = holstein_2site_1phonon_matrix(t, omega, g, Nph)
evals, _ = spla.eigsh(H, k=6, which="SA")
evals = np.sort(evals.real)
print("Lowest eigenvalues:", evals)
