import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Define elements and minimal basis set 
symbols = ["Li", "H"]
basis_set = "sto-3g"
# The bond lengths to sweep
r_values = np.arange(1.0, 5.0, 0.25)

energies_hf = []
energies_fci = []
energies_vqe = [] 

for r in r_values:
    # Define geometry
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]])
    
    # 1. Build the Hamiltonian
    # This contains all the physics (kinetic + potential + repulsion)
    H, qubits = qml.qchem.molecular_hamiltonian(
        symbols, geometry, basis=basis_set
    )

    molecule = qml.qchem.Molecule(
    symbols,
    geometry,
    charge=0,
    mult=1,
    basis_name='sto-3g')
    
    # Define the initial many-particle wave function in the Hartree-Fock (HF) approximation, which is a Slater determinant
    # The HF state is just the first N electrons occupying the first N qubits.
    # LiH has 4 electrons.
    hf_state = qml.qchem.hf_state(electrons=4, orbitals=qubits)
    dev = qml.device("default.qubit", wires=qubits) # use the default for now
    
    @qml.qnode(dev)
    def calculate_hf_energy():
        qml.BasisState(hf_state, wires=range(qubits))
        return qml.expval(H)
    
    # If I assume electrons don't correlate and just occupy the lowest available energy slots,
    # what is the energy?
    e_hf = calculate_hf_energy()
    energies_hf.append(e_hf)

    ## Now run the VQE
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
    energies_vqe.append(prev_energy)


    # Calculate Full CI (Exact, but brute force :( ): Convert H to a matrix and diagonalize it to find the absolute lowest eigenvalue
    H_mat = qml.matrix(H)
    eigenvalues = np.linalg.eigvalsh(H_mat)
    e_fci = eigenvalues[0]
    energies_fci.append(e_fci)
    
    print(f"{r:<15.2f} | {e_hf:<15.5f} | {e_fci:<15.5f}")


plt.figure(figsize=(10, 6))
plt.plot(r_values, energies_hf, 'k--', label="Hartree-Fock (Classical)", linewidth=2)
plt.plot(r_values, energies_fci, 'b-', label="QVE result", linewidth=2, alpha=0.6)
plt.plot(r_values, energies_fci, 'b-', label="Full CI", linewidth=2, alpha=0.6)
plt.fill_between(r_values, energies_hf, energies_fci, color='gray', alpha=0.2, label="Correlation Energy")
plt.xlabel("Bond Length (Bohr)")
plt.ylabel("Energy (Hartree)")
plt.title("LiH Benchmark: Mean-Field vs. Exact Solution")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()