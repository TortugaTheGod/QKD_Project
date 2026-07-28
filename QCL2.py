import cirq
import sympy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random


signal = cirq.NamedQubit('signal')
eve = cirq.NamedQubit('eve')
sim = cirq.DensityMatrixSimulator()


target_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])

def get_fidelities(rho_4x4):
    rho = rho_4x4.reshape((2, 2, 2, 2))
    bob_rho = np.trace(rho, axis1=1, axis2=3)
    eve_rho = np.trace(rho, axis1=0, axis2=2)
    fab = cirq.fidelity(bob_rho, target_state)
    fae = cirq.fidelity(eve_rho, target_state)
    return fab, fae

def evaluate_circuit(circ, noise=0.0):
    circuit = cirq.Circuit()
    circuit.append(cirq.H(signal))
    circuit.append(circ)
    if noise > 0:
        circuit.append(cirq.depolarize(p=noise).on_each(signal, eve))

    result = sim.simulate(circuit, qubit_order=[signal, eve])
    return get_fidelities(result.final_density_matrix)

def build_pccm():
    theta=np.pi/4
    pccm = cirq.Circuit()
    pccm.append(cirq.rx(np.pi/2)(signal))
    pccm.append(cirq.ry(theta)(eve).controlled_by(signal))
    pccm.append(cirq.ry(-np.pi)(signal).controlled_by(eve))
    pccm.append(cirq.rx(-np.pi/2)(signal))
    pccm.append(cirq.rx(-np.pi/2)(eve))
    return pccm



symbols = sympy.symbols('theta0:18')

def build_qcl_ansatz():

    ansatz = cirq.Circuit()
    start = 0
    for i in range(2):
        ansatz.append(cirq.rx(symbols[start + 0])(signal))
        ansatz.append(cirq.ry(symbols[start + 1])(signal))
        ansatz.append(cirq.rz(symbols[start + 2])(signal))
        ansatz.append(cirq.rx(symbols[start + 3])(eve))
        ansatz.append(cirq.ry(symbols[start + 4])(eve))
        ansatz.append(cirq.rz(symbols[start + 5])(eve))
        ansatz.append(cirq.CNOT(signal, eve))
        start += 6
    """
    if (x basis) :
        ansatz.append(cirq.rx(symbols[12])(eve))
        ansatz.append(cirq.ry(symbols[13])(eve))
        ansatz.append(cirq.rz(symbols[14])(eve))
    else :
        ansatz.append(cirq.rx(symbols[15])(eve))
        ansatz.append(cirq.ry(symbols[16])(eve))
        ansatz.append(cirq.rz(symbols[17])(eve))
    """
    return ansatz

ansatz_circuit = build_qcl_ansatz()
target = random.uniform(0.5, 1.0)
def qcl_loss(weights, noise):
    resolver = dict(zip(symbols, weights))
    resolved_circuit = cirq.resolve_parameters(ansatz_circuit, resolver)
    fab, fae = evaluate_circuit(resolved_circuit, noise)

    loss = 10*(fab - target) - fae 
    return loss

def train_qcl(noise):
    
    initial_weights = np.array([0, 0, 0, 0, 0, 0])
    result = minimize(qcl_loss, initial_weights, args=(noise), method='COBYLA',options={'maxiter': 500})
    
    resolver = dict(zip(symbols, result.x))
    resolved_circuit = cirq.resolve_parameters(ansatz_circuit, resolver)
    return evaluate_circuit(resolved_circuit, noise)


noise = np.linspace(0.0, 0.2, 10) 
