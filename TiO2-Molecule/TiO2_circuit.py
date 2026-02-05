
import pennylane as qml
from pennylane import qchem
import pickle
import optax
import os 
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
from jax import numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)

symbols = ['Ti', 'O', 'O']

geometry = jnp.array([
    [0.000, 0.000, 0.000],   # Ti
    [1.620, 0.000, 0.000],   # O
    [-0.810, 0.000, 1.403],  # O (this is roughly 112 degrees)
])

# Assume that your Ti-O-Ti molecule is on the 
# Charge=0, Multiplicity=1 (Singlet) are defaults
molecule = qml.qchem.Molecule(symbols, geometry, load_data=True) # creates 12 spin-orbitals, 43 basis functions
print("The number of atomic basis functions per atom: ", molecule.n_basis) 
orbitals = sum(molecule.n_basis) # Only valid fpr the simplest basis: total number of orbitals for the
print("The number of atomic basis functions: ", orbitals) # relevant for HF
# molecule = qml.qchem.Molecule(symbols, geometry, basis_name="6-31G", load_data=True) # creates 22 spin-orbitals. 67 basis functions
electrons = 38 # Total electrons in TiO2 molecule
# The number of orbitals depend on how many molecular orbitals come out from the Hartree fock
active_electrons = 6  # 
active_orbitals = 6
qubits = 2*active_orbitals # equivalent to spin orbitals

folder = "TiO2-Molecule"
filepath = os.path.join(folder,"Pauli_MoleculeTiO2_JW.pkl" )
with open(filepath,"rb") as f:
    h_pauli = pickle.load(f)
    
# Hartree-Fock State (Must be 'occupation_number' for Jordan-Wigner)
hf_state = qchem.hf_state(active_electrons, qubits, basis="occupation_number")
""" energy = qchem.hf_energy(molecule)(geometry)
print(f"Hartree-Fock energy: {energy:.6f} Ha") """

# Generate Excitations for UCCSD
singles, doubles = qchem.excitations(active_electrons, qubits)
# Map excitations to the wires the UCCSD circuit will act on
s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

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
theta = params 
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


