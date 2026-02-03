from pennylane import qchem
from jax import numpy as jnp
import jax
import pennylane as qml
from pennylane.fermi import from_string
import optax

import optax


jax.config.update("jax_enable_x64", True)
# We start by defining the geometry of the LiH molecule
# Bond length is approx. 3 Bohr
symbols = ['Li', 'H']
geometry = jnp.array([[0.0, 0.0, 0.0],
                      [0.0, 0.0, 3.0]])

# Charge=0, Multiplicity=1 (Singlet) are defaults
mol = qchem.Molecule(symbols, geometry)
# Let´s start by first building the fermionic Hamiltonian: 
# 1. Get Fermionic Hamiltonian (e.g. from integrals)
h_fermi = qchem.fermionic_hamiltonian(mol)()
electrons = 3
active_electrons = electrons
qubits = len(h_fermi.wires)
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


