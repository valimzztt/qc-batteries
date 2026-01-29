import pennylane as qml
from pennylane import numpy as np

def build_holstein_hamiltonian(t, omega, g, max_phonons=8):
    """
    Constructs the matrix for a 2-site Holstein Polaron model.
    
    Args:
        t (float): Hopping integral (kinetic energy).
        omega (float): Phonon frequency (stiffness of the O-bond).
        g (float): Coupling strength (how hard the electron pushes).
        max_phonons (int): Truncation for the phonon Hilbert space.
    """
    
    # We define the operators in their own subspaces
    # Electronic Basis: |L> = [1,0], |R> = [0,1]
    # Hopping moves electron L <-> R
    # H_el = -t * sigma_x
    H_hopping_el = -t * np.array([[0, 1], 
                                  [1, 0]])
    
    # Operator that checks WHERE the electron is (Polarization)
    # (n_R - n_L) corresponds to sigma_z: [[-1, 0], [0, 1]] 
    # (assuming |L> is -1 and |R> is +1 logic)
    n_diff_el = np.array([[-1, 0], 
                          [0, 1]])
    
    # Phonon Basis: Fock states |0>, |1>, ... |M>
    # Creation operator a_dagger
    a_dag = np.diag(np.sqrt(np.arange(1, max_phonons)), k=-1)
    a = a_dag.T
    
    # Number operator a^dagger a
    n_ph = np.diag(np.arange(max_phonons))
    
    # Position operator (x ~ a^dagger + a)
    x_ph = a_dag + a
    
    # --- 2. Combine subspaces using Kronecker Product (Tensor Product) ---
    # Identity matrices for the subspace not being acted on
    I_el = np.eye(2)
    I_ph = np.eye(max_phonons)
    
    # Full terms: H = H_el (x) I_ph  +  I_el (x) H_ph  +  H_coup
    
    # Term 1: Kinetic Energy (Electron hopping, phonon does nothing)
    H_kin = np.kron(H_hopping_el, I_ph)
    
    # Term 2: Lattice Energy (Electron does nothing, phonon vibrates)
    H_vib = omega * np.kron(I_el, n_ph)
    
    # Term 3: Interaction (Electron position shifts phonon position)
    H_int = g * omega * np.kron(n_diff_el, x_ph)
    
    # Total Matrix
    H_total = H_kin + H_vib + H_int
    
    return H_total

# This is just a toy model to see whether this thinking makes sense
# Parameters for FePO4 (approximate relative scales)
t_val = 0.1   # Hopping is 'slow' (100 meV)
w_val = 0.05  # Phonon is 'stiff' (50 meV)
g_val = 2.0   # Coupling is STRONG (This creates the polaron)

H_matrix = build_holstein_hamiltonian(t_val, w_val, g_val)
print(H_matrix)

# Define a PennyLane device
# We need ceil(log2(16)) = 4 qubits to represent this 16x16 matrix
num_qubits = int(np.ceil(np.log2(H_matrix.shape[0])))
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev)
def holstein_vqe(params):
    # Use a basic ansatz to find the ground state of this toy matrix
    qml.ArbitraryStatePreparation(params, wires=range(num_qubits))
    return qml.expval(qml.Hermitian(H_matrix, wires=range(num_qubits)))

# Optimization
params = np.random.random(2**(num_qubits + 1) - 2) # simple ansatz for full state vector
opt = qml.GradientDescentOptimizer(stepsize=0.1)

print("Training Holstein Toy Model...")
for i in range(10000):
    params, energy = opt.step_and_cost(holstein_vqe, params)
    if i % 5 == 0:
        print(f"Step {i}: Energy = {energy:.4f} eV")

print(f"Final Ground State Energy: {energy:.4f} eV") 