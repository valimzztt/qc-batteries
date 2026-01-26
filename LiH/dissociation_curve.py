import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# We are interested in LiH
symbols = ["Li", "H"]
# We define the bond length up to 5 (broken bond)
bond_lengths = np.arange(1.0, 5.0, 0.25)
energies = []

# this is the VQE solver function 
def solve_ground_state(r):
    # At each step, we define a new geoemtry based on r
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]])
    # Build Hamiltonian (Active Space: 2 electrons, 2 orbitals -> 4 qubits)
    # active space is large enough to capture bond breaking
    H, qubits = qml.qchem.molecular_hamiltonian(
        symbols, geometry, charge=0, mult=1,
        active_electrons=2, active_orbitals=2
    )
    
    dev = qml.device("default.qubit", wires=qubits)
    # Define the ansatz (Double Excitation handles static correlation well)
    hf_state = qml.qchem.hf_state(2, qubits)
    
    @qml.qnode(dev)
    def cost_fn(theta):
        qml.BasisState(hf_state, wires=range(qubits))
        qml.DoubleExcitation(theta, wires=[0, 1, 2, 3])
        return qml.expval(H)
    
    # Optimization
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    theta = np.array(0.0, requires_grad=True)
    
    # This is the optimization loop that we use to minimize the cost function, optimizing the variational parameter 
    for n in range(20):
        theta, prev_energy = opt.step_and_cost(cost_fn, theta)
        
    return prev_energy

# We run the scan, which effectively simulates the Reaction Path
for r in bond_lengths:
    e = solve_ground_state(r)
    energies.append(e)
    print(f"Bond Length: {r:.2f} Bohr | Energy: {e:.5f} Ha")

# The equilibrium energy is the minimum 
e_eq =min(energies)
# energy when atoms are far apart
e_dis = energies[-1]

# Bond dissociation energy
bond_energy = e_dis - e_eq

# Equilibrium bond length
idx = energies.index(e_eq)
bond_length = bond_lengths[idx]

print(f"The equilibrium bond length is {bond_length:.3f} Bohrs")
print(f"The bond dissociation energy is {bond_energy:.6f} Hartrees")

# We plot the energy versus bond length plot, which tells us
# 1. Depth =Bond Strength
# 2. Dissociation limit from the tail (where DFT often fails)
plt.figure(figsize=(10, 6))
plt.plot(bond_lengths, energies, 'o-', linewidth=2, color='crimson')
plt.title("The Bond Dissociation curve (Quantum VQE)")
plt.xlabel("Bond Distance (Bohr)")
plt.ylabel("Energy (Hartree)")
plt.grid(True, alpha=0.3)
plt.show()