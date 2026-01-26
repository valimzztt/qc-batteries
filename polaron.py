import pennylane as qml
from pennylane import numpy as np

# -----------------------------
# Model parameters (toy values)
# -----------------------------
t = 0.15      # hopping (eV)
omega = 0.05  # phonon energy (eV)
g = 0.20      # e-ph coupling (eV)
# We'll scan a bias "Delta" to mimic moving along a reaction coordinate.
# Physically, Delta can stand in for an external/local environment shift.

n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

def holstein_hamiltonian(Delta):
    """
    2-qubit Holstein dimer with truncated phonon.
    Qubit 0 = electron site (sigma_z gives site polarization)
    Qubit 1 = phonon 0/1
    """
    # Electron part: bias + hopping
    # H_e = (Delta/2) * sigma_z(e) + t * sigma_x(e)
    H_e = (Delta/2) * qml.PauliZ(0) + t * qml.PauliX(0)

    # Phonon part: H_ph = omega * n = omega * (I - Z)/2
    H_ph = (omega/2) * (qml.Identity(1) - qml.PauliZ(1))

    # Coupling: H_c = g * (sigma_z(e)) * (a + a^\dagger)
    # with (a + a^\dagger) ≈ X on truncated phonon qubit
    H_c = g * qml.PauliZ(0) @ qml.PauliX(1)

    return H_e + H_ph + H_c


# This is a very simple ansatz
def ansatz(params):
    # params shape: (3,) just to keep it small
    qml.RY(params[0], wires=0)
    qml.RZ(params[1], wires=0)

    qml.RY(params[2], wires=1)

    # entangle electron <-> phonon
    qml.CNOT(wires=[0, 1])

@qml.qnode(dev, interface="autograd")
def energy_qnode(params, Delta):
    ansatz(params)
    H = holstein_hamiltonian(Delta)
    return qml.expval(H)

def run_vqe(Delta, steps=250, lr=0.08, seed=0):
    rng = np.random.default_rng(seed)
    params = rng.normal(scale=0.1, size=(3,), requires_grad=True)

    opt = qml.GradientDescentOptimizer(lr)
    for _ in range(steps):
        params = opt.step(lambda p: energy_qnode(p, Delta), params)

    E = energy_qnode(params, Delta)
    return float(E), params


# -----------------------------
# Scan "reaction coordinate"
# -----------------------------
Deltas = np.linspace(-0.6, 0.6, 25)

Egs = []
params_warm = None

for i, Delta in enumerate(Deltas):
    # Warm-start: reuse previous optimum to speed up convergence
    if params_warm is None:
        E, params_warm = run_vqe(Delta, steps=350, lr=0.08, seed=42)
    else:
        # quick refinement from previous params
        params = params_warm.copy()
        opt = qml.GradientDescentOptimizer(0.05)
        for _ in range(150):
            params = opt.step(lambda p: energy_qnode(p, Delta), params)
        E = float(energy_qnode(params, Delta))
        params_warm = params

    Egs.append(E)

Egs = np.array(Egs)

# crude barrier: max - min over the scan (toy proxy)
Ea_proxy = float(Egs.max() - Egs.min())

print("Deltas:", Deltas)
print("E_ground(Delta):", Egs)
print(f"Proxy activation barrier (max-min over scan): {Ea_proxy:.4f} eV")