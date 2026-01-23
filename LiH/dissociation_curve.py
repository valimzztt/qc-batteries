import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define the Molecule and Active Space ---
# We simulate LiH bond breaking.
symbols = ["Li", "H"]

# We sweep the bond length 'r' from equilibrium (1.5) to broken (5.0)
bond_lengths = np.arange(1.0, 5.0, 0.25)
energies = []

# --- 2. The VQE Solver Function ---
def solve_ground_state(r):
    # Geometry for this step
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]])
    
    # Build Hamiltonian (Active Space: 2 electrons, 2 orbitals -> 4 qubits)
    # This keeps simulation fast but captures the bond breaking physics
    H, qubits = qml.qchem.molecular_hamiltonian(
        symbols, geometry, charge=0, mult=1,
        active_electrons=2, active_orbitals=2
    )
    
    dev = qml.device("default.qubit", wires=qubits)
    
    # Define Ansatz (Double Excitation handles static correlation well)
    hf_state = qml.qchem.hf_state(2, qubits)
    
    @qml.qnode(dev)
    def cost_fn(theta):
        qml.BasisState(hf_state, wires=range(qubits))
        qml.DoubleExcitation(theta, wires=[0, 1, 2, 3])
        return qml.expval(H)
    
    # Optimization
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    theta = np.array(0.0, requires_grad=True)
    
    # Quick optimization loop
    for n in range(20):
        theta, prev_energy = opt.step_and_cost(cost_fn, theta)
        
    return prev_energy

# --- 3. Run the Scan (The "Reaction Path") ---
print("Simulating Bond Breaking (SEI Formation Proxy)...")
for r in bond_lengths:
    e = solve_ground_state(r)
    energies.append(e)
    print(f"Bond Length: {r:.2f} Bohr | Energy: {e:.5f} Ha")


# equilibrium energy
e_eq = min(energies)
# energy when atoms are far apart
e_dis = energies[-1]

# Bond dissociation energy
bond_energy = e_dis - e_eq

# Equilibrium bond length
idx = energies.index(e_eq)
bond_length = bond_lengths[idx]

print(f"The equilibrium bond length is {bond_length:.1f} Bohrs")
print(f"The bond dissociation energy is {bond_energy:.6f} Hartrees")

# --- 4. Plotting ---
# The shape of this curve tells us:
# 1. Depth = Bond Strength
# 2. Curvature = Vibrational Frequency
# 3. Tail = Dissociation Limit (where DFT often fails)
plt.figure(figsize=(10, 6))
plt.plot(bond_lengths, energies, 'o-', linewidth=2, color='crimson')
plt.title("Bond Dissociation Profile (Quantum VQE)")
plt.xlabel("Bond Distance (Bohr)")
plt.ylabel("Energy (Hartree)")
plt.grid(True, alpha=0.3)
plt.show()

# Note that these estimates can be improved
# by using bigger basis sets and extrapolating to the complete basis set limit 