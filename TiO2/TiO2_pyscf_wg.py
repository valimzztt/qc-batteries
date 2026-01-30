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

# Initial state (HF) in the Jordan Wigner basis
hf_state = qchem.hf_state(active_electrons, qubits, basis="occupation_number") 
dev = qml.device("default.qubit", wires=qubits)
@qml.qnode(dev, interface="jax")
def circuit(param, wires):
    qml.BasisState(hf_state, wires=wires)
    qml.DoubleExcitation(param, wires=[0, 1, 2, 3])
    return qml.expval(h_pauli)
def cost_fn(param):
    return circuit(param, wires=range(qubits))
import optax

max_iterations = 100
conv_tol = 1e-06

opt = optax.sgd(learning_rate=0.4)
theta = np.array(0.)

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

    conv = np.abs(energy[-1] - energy[-2])

    if n % 2 == 0:
        print(f"Step = {n},  Energy = {energy[-1]:.8f} Ha")

    if conv <= conv_tol:
        break

print("\n" f"Final value of the ground-state energy = {energy[-1]:.8f} Ha")
print("\n" f"Optimal value of the circuit parameter = {angle[-1]:.4f}")