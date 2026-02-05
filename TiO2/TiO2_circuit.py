
import pennylane as qml
from pennylane import qchem
from jax import numpy as jnp
import jax
import pickle
#jax.config.update('jax_num_cpu_devices', 8)
jax.config.update("jax_enable_x64", True)

# 1. Define TiO2 Dimer Geometry (Ti-O-Ti bridge)
symbols = ['Ti', 'O', 'Ti']
""" These are the coordinates from MP:
(3.2, 3.2, 4.439) 
(2.3, 2.3, 2.959)
(3.2, 3.2, 1.480) """

# Coordinates taken from MP
geometry = jnp.array([[3.2, 3.2, 4.439],
                      [2.3, 2.3, 2.959],
                      [3.2, 3.2, 1.480]])

# We Build the molecule
mol = qchem.Molecule(symbols, geometry, mult=1, load_data=True)

with open("TiO2_Hamiltonian_pyscf.pkl", "rb") as f:
    H_pauli = pickle.load(f)
    
# 3. Active Space Selection
# Titanium (STO-3G) has many orbitals. To keep qubits at ~12:
# We select 1 active electron (the polaron) and 6 active orbitals 
# (focused on the Ti 3d and O 2p hybridization path)
active_electrons = 2
active_orbitals = 6
# We now build the quantum circuit with  the UCCSD ansatz: which is constructed with a se of single and double 
# excitation operators. In Pennylane, SingleExcitation and DoubleExcitation operators are efficient but only
# compatible with the Jordan-Wigner mapping. 
# We need the initial state that has the correct number of electrons. 
# We use the Hartree-Fock state which can be obtained in a user-defined basis 
# For that, we need to specify the number of electrons, the number of orbitals and the desired mapping.
# Expected: 10 qubits (5 orbitals * 2 spin states)
# Define active space
electrons = 52 # Total electrons in TiO2 molecule
orbitals = 35
core_indices, active_indices= qchem.active_space(electrons, orbitals, active_electrons=2, active_orbitals=6)
print("Core orbitals:", core_indices)
print("Active orbitals:", active_indices)

# Transform the fermionic Hamiltonian to its qubit representation via Bravji-Kitaev transformation
h_pauli = qml.bravyi_kitaev(h_fermi, qubits, tol=1e-16)
h_pauli = qml.jordan_wigner(h_fermi )

# We use the Hartree-Fock state which can be obtained in a user-defined basis 
# For that, we need to specify the number of electrons, the number of orbitals and the desired mapping.
hf_state = qchem.hf_state(electrons, qubits, basis="bravyi_kitaev") 
hf_state = qchem.hf_state(electrons, qubits, basis="occupation_number") 

# Hartree-Fock State (Must be 'occupation_number' for Jordan-Wigner)
hf_state = qchem.hf_state(active_electrons, qubits, basis="occupation_number")

# Generate Excitations for UCCSD
singles, doubles = qchem.excitations(active_electrons, qubits)
# Map excitations to the wires the UCCSD circuit will act on
s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

# Define the Device
dev = qml.device("default.qubit", wires=qubits)

# we define the node (what is the meaning of this?)
@qml.qnode(dev, interface="jax")
def circuit(params, wires, s_wires, d_wires, hf_state):
    print("We are using the following HF state", hf_state)
    qml.UCCSD(params, wires, s_wires, d_wires, hf_state)
    return qml.expval(h_pauli)

# Define the initial values of the circuit parameters
params = jnp.zeros(len(singles) + len(doubles))


def cost_fn(param):
    result =  circuit(param, wires=range(qubits), s_wires=s_wires, d_wires=d_wires, hf_state=hf_state)
    print(result)
    return jnp.real(result)

max_iterations = 100
conv_tol = 1e-06

opt = optax.sgd(learning_rate=0.4)
theta = params # the parameters to optimize are the excitations 
# store the values of the cost function
energy = [cost_fn(theta)]

# store the values of the circuit parameter
angle = [theta]
opt_state = opt.init(theta)
for n in range(max_iterations):

    gradient = jax.grad(cost_fn)(theta)
    updates, opt_state = opt.update(gradient, opt_state)
    theta = optax.apply_updates(theta, updates)

    angle.append(theta)
    energy.append(cost_fn(theta))

    conv = jnp.abs(energy[-1] - energy[-2])

    if n % 2 == 0:
        print(f"Step = {n},  Energy = {energy[-1]:.8f} Ha")

    if conv <= conv_tol:
        break

print("\n" f"Final value of the ground-state energy = {energy[-1]:.8f} Ha")
print("\n" f"Optimal value of the circuit parameter = {angle[-1]:.4f}")
