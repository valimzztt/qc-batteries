import pennylane as qml
from pennylane import numpy as np

# Physical parameters for LFP (approximated)
t = 0.15     # Hopping integral (eV)
g = 0.45     # Electron-phonon coupling (eV) - strong for polaron
omega = 0.05 # Phonon frequency (eV)

import pennylane as qml
from pennylane import numpy as np

# --- LiFePO4 Specific Constants (from DFT+U literature) ---
# Values in eV
T_HOP = 0.05       # Small tunneling due to Fe-O-Fe bond angles
LAMBDA = 0.50      # Reorganization energy (approx 2*E_pol)
OMEGA_PH = 0.045   # Average frequency of the Fe-O breathing mode
U_HUBBARD = 4.5    # Strong on-site repulsion
V_VACANCY = 0.15   # Potential shift due to a nearby Lithium vacancy

# Derived coupling g: lambda = 2 * g^2 / omega
G_COUPLING = np.sqrt(LAMBDA * OMEGA_PH / 2)

def build_accurate_lfp_hamiltonian(xi, include_vacancy=True):
    """
    Constructs an LFP-specific Hamiltonian.
    xi: Reaction coordinate [-1, 0]
    include_vacancy: Adds the Li+ vacancy trap effect.
    """
    coeffs = []
    ops = []

    # 1. Electronic Hopping along b-axis
    for i, j in [(0, 1), (1, 2)]:
        coeffs.extend([-T_HOP/2, -T_HOP/2])
        ops.extend([qml.PauliX(i) @ qml.PauliX(j), qml.PauliY(i) @ qml.PauliY(j)])

    # 2. Site Energies + Vacancy Effect
    # At xi=-1, the system is at Site 0.
    # If a vacancy is at Site 0, it lowers the energy there.
    bias_0 = xi * (LAMBDA / 2)
    bias_1 = -xi * (LAMBDA / 2)
    
    if include_vacancy:
        # The vacancy stabilizes the polaron at its starting site (Site 0)
        bias_0 -= V_VACANCY 
        
    coeffs.extend([bias_0, bias_1])
    ops.extend([qml.PauliZ(0), qml.PauliZ(1)])

    # 3. Phonon and Interaction Terms
    X_mat = np.array([[0, 1, 0, 0], [1, 0, np.sqrt(2), 0], 
                      [0, np.sqrt(2), 0, np.sqrt(3)], [0, 0, np.sqrt(3), 0]])
    
    for i in range(3):
        # Electron-Phonon coupling (The "Small Polaron" force)
        ph_qubits = [3+2*i, 4+2*i]
        coeffs.append(G_COUPLING)
        ops.append(qml.Projector([1], wires=i) @ qml.Hermitian(X_mat, wires=ph_qubits))
        
        # Local Phonon Energy
        coeffs.append(OMEGA_PH)
        # Simplified Number operator n = b+ b
        n_mat = np.diag([0, 1, 2, 3])
        ops.append(qml.Hermitian(n_mat, wires=ph_qubits))

    return qml.Hamiltonian(coeffs, ops)

# --- VQE Execution Setup ---
dev = qml.device("default.qubit", wires=9)

@qml.qnode(dev)
def cost_fn(params, H):
    # Use a basic hardware-efficient ansatz for the 9 qubits
    qml.StronglyEntanglingLayers(params, wires=range(9))
    return qml.expval(H)

import matplotlib.pyplot as plt

# 1. Define the range for the reaction coordinate
# We only need to scan from -1 (Site 1) to 0 (Transition State)
xi_steps = np.linspace(-1.0, 0.0, 15)
energies = []

# 2. Setup the ansatz shape (3 layers of entangling gates for 9 qubits)
n_layers = 3
params_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=9)
# Initialize parameters randomly
params = np.random.uniform(low=0, high=2 * np.pi, size=params_shape, requires_grad=True)

# 3. Optimization Loop
opt = qml.AdamOptimizer(stepsize=0.1)

print(f"{'xi':>10} | {'Ground Energy (eV)':>20}")
print("-" * 35)

for xi in xi_steps:
    # Build Hamiltonian for current reaction coordinate
    H = build_accurate_lfp_hamiltonian(xi)
    
    # Run VQE optimization for this point
    # We do fewer steps per xi because we "warm-start" from previous params
    for _ in range(100):
        params = opt.step(lambda p: cost_fn(p, H), params)
    
    energy = cost_fn(params, H)
    energies.append(energy)
    print(f"{xi:10.2f} | {energy:20.5f}")

# 4. Calculate Activation Energy
E_initial = energies[0]
E_transition = energies[-1]
E_a = E_transition - E_initial

print(f"\nCalculated Activation Energy (Ea): {E_a * 1000:.2f} meV")