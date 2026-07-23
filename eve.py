import numpy as np
import cirq
from scipy.optimize import minimize # for cobyla

qubits = cirq.NamedQubit.range(2, prefix = 'q')
sim = cirq.Simulator()


# Prepare Alice state

def eve_unitary_circuit(circuit, theta, layers):
    start = 0
    for i in range(layers): # for n layers, appends rx, ry, rz for qubit 0 and 1, then CNOT gate
        circuit.append(cirq.rx(theta[start + 0]).on(qubits[0]))
        circuit.append(cirq.ry(theta[start + 1]).on(qubits[0]))
        circuit.append(cirq.rz(theta[start + 2]).on(qubits[0]))

        circuit.append(cirq.rx(theta[start + 3]).on(qubits[1]))
        circuit.append(cirq.ry(theta[start + 4]).on(qubits[1]))
        circuit.append(cirq.rz(theta[start + 5]).on(qubits[1]))

        circuit.append(cirq.CNOT(qubits[0], qubits[1]))
        idx += 6


# eve V(Λ), i don't know what this is
