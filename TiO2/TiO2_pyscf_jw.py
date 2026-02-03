import pennylane as qml
from pennylane import qchem
from pennylane.fermi import from_string
import jax
from jax import numpy as jnp
import optax
import numpy as np

jax.config.update('jax_enable_x64', True)
symbols = ['Ti', 'O', 'Ti']
geometry = jnp.array([[3.2, 3.2, 4.439],
                      [2.3, 2.3, 2.959],
                      [3.2, 3.2, 1.480]])

active_electrons = 2
active_orbitals = 6

# Jordan Wigner
h_pauli, qubits = qchem.molecular_hamiltonian(
    symbols, geometry, mult=1, basis="sto-3g",
    mapping = "jordan_wigner",
    method = "pyscf",
    active_electrons=active_electrons, active_orbitals=active_orbitals, load_data=True
)

import pickle
with open("TiO2_Hamiltonian_pyscf_jw.pkl", "wb") as f:
    pickle.dump(h_pauli, f)
with open("TiO2_Hamiltonian_pyscf_jw.pkl", "rb") as f:
    h_pauli = pickle.load(f) # in the JW basis 

# How many active electrons
active_electrons = 2

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


