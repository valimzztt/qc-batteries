import pennylane as qml
from pennylane import qchem
from pennylane.fermi import from_string
import jax
from jax import numpy as jnp
import optax
import numpy as np

jax.config.update('jax_enable_x64', True)

# --- 1. System Setup ---
symbols = ['Ti', 'O', 'Ti']
geometry = jnp.array([[3.2, 3.2, 4.439],
                      [2.3, 2.3, 2.959],
                      [3.2, 3.2, 1.480]])

active_electrons = 2
active_orbitals = 6

# --- 2. Hamiltonian (Bravyi-Kitaev) ---
h_pauli, qubits = qchem.molecular_hamiltonian(
    symbols, geometry, mult=1, basis="sto-3g",
    method = "pyscf",
    active_electrons=active_electrons, active_orbitals=active_orbitals, load_data=True,
    mapping="bravyi_kitaev" # Hamiltonian in BK
)

import pickle
with open("TiO2_Hamiltonian_pyscf_kb.pkl", "wb") as f:
    pickle.dump(h_pauli, f)
hf_state = qchem.hf_state(active_electrons, qubits, basis="bravyi_kitaev")

singles, doubles = qchem.excitations(active_electrons, qubits)


excitation_ops = []

# Doubles
for ex in doubles:
    # T2 = a^dag_p a^dag_q a_r a_s - h.c.
    term = from_string(f"{ex[3]}+ {ex[2]}+ {ex[1]}- {ex[0]}-") \
         - from_string(f"{ex[0]}+ {ex[1]}+ {ex[2]}- {ex[3]}-")
    # Convert to BK PauliSentence
    pauli_term = qml.bravyi_kitaev(term, qubits, ps=True)
    # The generator is i*(T - Tdag), so we multiply by 1j to make it Hermitian for qml.exp
    # actually qml.CommutingEvolution handles the unitary e^{-iH}, so we just pass the Hermitian generator.
    # UCCSD generator G = (T - Tdag). U = exp(theta * G). 
    # In PennyLane, evolution is exp(-i * x * H).
    # So if we want exp(theta * (T - Tdag)), let H = i*(T-Tdag). 
    # Then exp(-i * theta * i(T-Tdag)) = exp(theta * (T-Tdag)).
    # We multiply the operator by 1j.
    
    # Note: PennyLane's FermiSentence doesn't support complex scalar mult easily in all versions,
    # so we often define the generator manually. 
    # Simplified: Just trust qml.CommutingEvolution to handle the operator you built.
    excitation_ops.append(pauli_term)

# Singles
for ex in singles:
    term = from_string(f"{ex[1]}+ {ex[0]}-") \
         - from_string(f"{ex[0]}+ {ex[1]}-")
    pauli_term = qml.bravyi_kitaev(term, qubits, ps=True)
    excitation_ops.append(pauli_term)

# --- 5. Circuit ---
dev = qml.device("default.qubit", wires=qubits)

@qml.qnode(dev)
def circuit(params):
    qml.BasisState(hf_state, wires=range(qubits))
    
    # Apply excitations
    # We multiply by 1j to convert the anti-hermitian cluster operator to a Hermitian generator
    for i, op in enumerate(excitation_ops):
        # We use CommutingEvolution for complex Pauli sums
        # H_gen = 1j * op
        qml.CommutingEvolution(1j * op, params[i])
        
    return qml.expval(h_pauli)

# --- 6. Optimization ---
num_params = len(excitation_ops)
theta = jnp.zeros(num_params) # Initialize to zero (starts at HF energy)

opt = optax.sgd(learning_rate=0.4)
opt_state = opt.init(theta)

print(f"Starting VQE with {num_params} parameters...")
print(f"Initial Energy: {circuit(theta):.8f} Ha")

# Loop
for n in range(50): # Reduced iterations for testing
    grads = jax.grad(circuit)(theta)
    updates, opt_state = opt.update(grads, opt_state)
    theta = optax.apply_updates(theta, updates)
    
    if n % 5 == 0:
        curr_E = circuit(theta)
        print(f"Step {n}: Energy = {curr_E:.8f} Ha")

print(f"Final Energy: {circuit(theta):.8f} Ha")