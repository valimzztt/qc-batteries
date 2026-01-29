import pennylane as qml
from pennylane import qchem
from jax import numpy as jnp
import jax
jax.config.update('jax_enable_x64', True)
#jax.config.update('jax_num_cpu_devices', 8)

# 1. Define TiO2 Dimer Geometry (Ti-O-Ti bridge)
symbols = ['Ti', 'O', 'Ti']
""" These are the coordinates from MP:
(3.2, 3.2, 4.439) 
(2.3, 2.3, 2.959)
(3.2, 3.2, 1.480) """


# Ti-O distance is typically ~1.9 - 2.0 Angstroms
geometry = jnp.array([[3.2, 3.2, 4.439],
                      [2.3, 2.3, 2.959],
                      [3.2, 3.2, 1.480]])

# We Build the molecule
mol = qchem.Molecule(symbols, geometry, mult=1, load_data=True)

# 3. Active Space Selection
# Titanium (STO-3G) has many orbitals. To keep qubits at ~12:
# We select 1 active electron (the polaron) and 6 active orbitals 
# (focused on the Ti 3d and O 2p hybridization path)
active_electrons = 2
active_orbitals = 6

# Define active space
electrons = 52 # Total electrons in TiO2 molecule
orbitals = 35
core, active = qchem.active_space(electrons, orbitals, active_electrons=2, active_orbitals=6)
print("Core orbitals:", core)
print("Active orbitals:", active)


print("Starting the Hamiltonian")
# 4. Build Hamiltonian
h_pauli, qubits = qchem.molecular_hamiltonian(
    symbols,
    geometry,
    method = "pyscf",
    # charge=7, 
    mult=1,
    basis="sto-3g",
    active_electrons=active_electrons,
    load_data = True,
    active_orbitals=active_orbitals,
    mapping = "bravyi_kitaev"
)

import pickle
with open("TiO2_Hamiltonian_pyscf_bk.pkl", "wb") as f:
    pickle.dump(h_pauli, f)

mapping="bravyi_kitaev"
print("Active Space Orbitals indices:", mapping) 
# Note: 'mapping' isn't always returned directly by the helper, 
# but PySCF prints the active space analysis if you enable verbose output:
# We need the initial state that has the correct number of electrons. 
# We use the Hartree-Fock state which can be obtained in a user-defined basis 
# For that, we need to specify the number of electrons, the number of orbitals and the desired mapping.
hf_state = qchem.hf_state(active_electrons, qubits, basis="bravyi_kitaev") 
print(hf_state )
#hf_state = qchem.hf_state(active_electrons, qubits, basis="occupation_number") 
# We now build the quantum circuit with  the UCCSD ansatz: which is constructed with a se of single and double 
# excitation operators. In Pennylane, SingleExcitation and DoubleExcitation operators are efficient but only
# compatible with the Jordan-Wigner mapping. 

# Construct the excitation operator mapping manuallz
from pennylane.fermi import from_string

singles, doubles = qchem.excitations(active_electrons, qubits)

singles_fermi = []
for ex in singles:
    singles_fermi.append(from_string(f"{ex[1]}+ {ex[0]}-")
                       - from_string(f"{ex[0]}+ {ex[1]}-"))

doubles_fermi = []
for ex in doubles:
    doubles_fermi.append(from_string(f"{ex[3]}+ {ex[2]}+ {ex[1]}- {ex[0]}-")
                       - from_string(f"{ex[0]}+ {ex[1]}+ {ex[2]}- {ex[3]}-"))
    
singles_pauli = []
for op in singles_fermi:
    singles_pauli.append(qml.bravyi_kitaev(op, qubits, ps=True))

doubles_pauli = []
for op in doubles_fermi:
    doubles_pauli.append(qml.bravyi_kitaev(op, qubits, ps=True))

    params = jnp.array([0.22347661, 0.0, 0.0])

dev = qml.device("default.qubit", wires=qubits)
#dev = qml.device("lightning.qubit", wires=qubits)
@qml.qnode(dev)
def circuit(params):
    qml.BasisState(hf_state, wires=range(qubits))

    for i, excitation in enumerate(doubles_pauli):
        qml.exp((excitation * params[i] / 2).operation()), range(qubits)

    for j, excitation in enumerate(singles_pauli):
        qml.exp((excitation * params[i + j + 1] / 2).operation()), range(qubits)

    return qml.expval(h_pauli)

print('Energy =', circuit(params))


import optax
import numpy as np

def cost_fn(param):
    return circuit(param, wires=range(qubits))

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