import pennylane as qml
from pennylane import numpy as np

t = 0.1   # Hopping is 'slow' (100 meV)
omega = 0.05  # determines the stiffness of the phonon
g = 2.0   # electro-phonon coupling
max_phonons = 8
    
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

# Phonon Basis: define the Phock states
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

# Full terms: H = H_el(x)I_ph+I_el(x)H_ph+H_coup
# term 1: kinetic energy 
H_kin = np.kron(H_hopping_el, I_ph)
# term 2: lattice energy
H_vib = omega * np.kron(I_el, n_ph)
# term 3: should catch the interaction
H_int = g * omega * np.kron(n_diff_el, x_ph)
H_matrix = H_kin + H_vib + H_int

# We need (log2(16)) = 4 qubits to represent this 16x16 matrix
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