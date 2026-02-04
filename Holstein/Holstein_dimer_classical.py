import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


def boson_operators(Nph):
    data, rows, cols = [], [], []
    for n in range(1, Nph):
        rows.append(n - 1)
        cols.append(n)
        data.append(np.sqrt(n))
    a = sp.csr_matrix((data, (rows, cols)), shape=(Nph, Nph), dtype=np.complex128)
    adag = a.conjugate().T
    n_op = adag @ a
    return a, adag, n_op

def pauli_sparse():
    I = sp.identity(2, format="csr", dtype=np.complex128)
    X = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.complex128))
    Z = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=np.complex128))
    return I, X, Z

def holstein_dimer_matrix(t, Delta, omega, g, Nph):
    I2, X, Z = pauli_sparse()
    a, adag, n_op = boson_operators(Nph)
    Iph = sp.identity(Nph, format="csr", dtype=np.complex128)

    H = (
        t     * sp.kron(X,  Iph, format="csr") +
        Delta * sp.kron(Z,  Iph, format="csr") +
        omega * sp.kron(I2, n_op, format="csr") +
        g     * sp.kron(Z,  (a + adag), format="csr")
    )
    return H.tocsr()

def ground_energy_quantized(t, Delta, omega, g, Nph):
    H = holstein_dimer_matrix(t, Delta, omega, g, Nph)
    # lowest eigenvalue
    evals, _ = spla.eigsh(H, k=1, which="SA")
    return float(evals[0].real)

# ---------- Classical (spring) adiabatic surface ----------
def E_el(Q, t, Delta, g):
    # ground electronic energy of 2x2 Hamiltonian: t*sigma_x + (Delta + gQ)*sigma_z
    return -np.sqrt(t**2 + (Delta + g*Q)**2)

def E_classical(Q, t, Delta, omega, g):
    # spring energy in dimensionless coordinate Q
    return E_el(Q, t, Delta, g) + 0.5 * omega * Q**2

def ground_energy_classical(t, Delta, omega, g, Qmax=8.0, npts=4001):
    Q = np.linspace(-Qmax, Qmax, npts)
    E = E_classical(Q, t, Delta, omega, g)
    i0 = np.argmin(E)
    return float(E[i0]), float(Q[i0]), Q, E

t     = 0.03
Delta = 0.00
omega = 0.05
g     = 0.20

# Classical minimum
Ecl, Q0, Qgrid, Egrid = ground_energy_classical(t, Delta, omega, g)
print(f"Classical min energy: {Ecl:.6f} eV at Q = {Q0:.3f}")

# Quantized ground energies vs Nph
for Nph in [2, 4, 6, 8, 10, 12, 16]:
    Eq = ground_energy_quantized(t, Delta, omega, g, Nph)
    print(f"Nph={Nph:2d}: Quantized E0 = {Eq:.6f} eV  |  (Eq - Ecl) = {Eq - Ecl:+.6f} eV")

# Plot classical surface
plt.figure(figsize=(7,4))
plt.plot(Qgrid, Egrid)
plt.axvline(Q0, linestyle="--")
plt.title("Classical (spring) adiabatic energy surface")
plt.xlabel("Dimensionless distortion coordinate Q")
plt.ylabel("Energy (eV)")
plt.grid(True)
plt.savefig("Coordinate_Q.png")
