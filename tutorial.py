import pennylane as qml

def my_quantum_function(x, y):
    qml.RZ(x, wires=0)
    qml.CNOT(wires=[0,1])
    qml.RY(y, wires=1)
    return qml.expval(qml.Z(wires=1))

dev = qml.device('default.qubit', wires=2)


import numpy as np

circuit = qml.QNode(my_quantum_function, dev)

import matplotlib.pyplot as plt
qml.drawer.use_style("black_white")
fig, ax = qml.draw_mpl(circuit)(np.pi/4, 0.7)
plt.savefig("Quantum_circuit.png")