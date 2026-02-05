import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def boson_ops(Nph):
    """
    This function creates Bosonic operators for a truncated Fock space of size Nph.
    """
    data, rows, cols = [], [], []
    for n in range(1, Nph):
        rows.append(n-1)
        cols.append(n)
        data.append(np.sqrt(n))
    a = sp.csr_matrix((data, (rows, cols)), shape=(Nph, Nph), dtype=np.complex128)
    adag = a.conjugate().T
    n = adag @ a
    I = sp.identity(Nph, format="csr", dtype=np.complex128)
    return a, adag, n, I

def electron_ops():
    """
    Creates Electronic operators for a single site (Qubit basis).
    Basis: |0> (Empty), |1> (Occupied)
    Returns: n (number operator), c (annihilation), c_dag (creation), I (identity)
    """
    # |0> = [1, 0], |1> = [0, 1]
    # Number operator n = |1><1|
    n = sp.csr_matrix([[0, 0], [0, 1]], dtype=np.complex128)
    
    # Annihilation c: c|1> = |0> -> [[0, 1], [0, 0]]
    c = sp.csr_matrix([[0, 1], [0, 0]], dtype=np.complex128)
    c_dag = c.conjugate().T
    
    I = sp.identity(2, format="csr", dtype=np.complex128)
    
    return n, c, c_dag, I

def holstein_2site_matrix(t, omega, g, Nph):
    """
    Constructs the full Hamiltonian for 2 Electronic Sites + 2 Local Phonon Modes.
    Structure: El1 (x) El2 (x) Ph1 (x) Ph2
    """
    # 1. Get Local Operators
    n_el, c, c_dag, I_el = electron_ops()
    a, adag, n_ph, I_ph = boson_ops(Nph)
    
    # Displacement Operator (a + a^dag)
    x_ph = a + adag
    # we build the full operators by using the Kronecker deltas
    # This is the electronic part: hopping
    # H_el = -t * (c1^d c2 + c2^d c1)
    # Term 1: c1^d (x) c2 (x) I (x) I
    h1 = sp.kron(c_dag, sp.kron(c, sp.kron(I_ph, I_ph)))
    # Term 2: c1 (x) c2^d (x) I (x) I
    h2 = sp.kron(c, sp.kron(c_dag, sp.kron(I_ph, I_ph)))
    H_el = -t * (h1 + h2)
    print("Electronic Hamiltonian")
    print(H_el)
    # H_ph = omega * (n_ph1 + n_ph2)
    # Term 1: I (x) I (x) n_ph1 (x) I
    ph1 = sp.kron(I_el, sp.kron(I_el, sp.kron(n_ph, I_ph)))
    print("One phonon matrix")
    print(ph1)
    # Term 2: I (x) I (x) I (x) n_ph2
    ph2 = sp.kron(I_el, sp.kron(I_el, sp.kron(I_ph, n_ph)))
    H_ph = omega * (ph1 + ph2)
    # This is the interaction part
    # H_int = g * [ n_el1 * (a1^d + a1) + n_el2 * (a2^d + a2) ]
    # Term 1: n_el1 (x) I (x) x_ph1 (x) I
    int1 = sp.kron(n_el, sp.kron(I_el, sp.kron(x_ph, I_ph)))
    #Term2: I (x) n_el2 (x) I (x) x_ph2
    int2 = sp.kron(I_el, sp.kron(n_el, sp.kron(I_ph, x_ph)))
    H_int = g * (int1 + int2)
    H_total = H_el + H_ph + H_int
    return H_total

t = 1.0 # this is the hopping integral
omega = 0.5 # phonon frequency 
g = 0.5 # electron-phonon coupling strength 
Nph = 4 # we set it to 2 to match the 'Unary' qubit mapping (0 or 1 phonon)

# We start by building the 2 site Holstein matrix
H = holstein_2site_matrix(t, omega, g, Nph)
print(f"2-site Hamiltonian", H)
print(f"Hamiltonian Matrix Shape: {H.shape}")
# 2(el)*2(el)*2(ph)*2(ph)=16
# Find the fist 6 eigenvalues by diagonalizing the matrix 
evals, evecs = spla.eigsh(H, k=6, which="SA")
evals = np.sort(evals.real)
print("Lowest eigenvalues")
print(evals)
print(f"The exact ground state (diagonalization): {evals[0]:.6f}")
