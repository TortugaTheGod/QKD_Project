import cirq
import sympy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random


signal = cirq.NamedQubit('signal')
eve = cirq.NamedQubit('eve')
sim = cirq.DensityMatrixSimulator(dtype=np.complex128)


target_state = [np.array([1/np.sqrt(2), 1/np.sqrt(2)]),np.array([1/np.sqrt(2),-(1/np.sqrt(2))]),np.array([0,1]),np.array([1,0])]



def get_fidelities(rho_4x4, idx):
    rho = rho_4x4.reshape((2, 2, 2, 2))
    bob_rho = np.trace(rho, axis1=1, axis2=3)
    eve_rho = np.trace(rho, axis1=0, axis2=2)
    
    target_vec = target_state[idx]
    
    fab = np.real(target_vec.conj() @ bob_rho @ target_vec)
    fae = np.real(target_vec.conj() @ eve_rho @ target_vec)
    return fab, fae

def evaluate_circuit(circ, noise=0.0):
    total_fab, total_fae = 0.0, 0.0
    num_states = len(target_state) 
    for idx in range(num_states):
        circuit = cirq.Circuit()

        if idx == 0:
            circuit.append(cirq.H(signal))
        elif idx == 1:
            circuit.append(cirq.X(signal))         
            circuit.append(cirq.H(signal))
        elif idx == 2:
            circuit.append(cirq.X(signal))
        elif idx ==3:
          circuit.append(cirq.I(signal))
        circuit.append(circ)
        if noise > 0:
            circuit.append(cirq.depolarize(p=noise).on_each(signal, eve))
        result = sim.simulate(circuit, qubit_order=[signal, eve])
        fab, fae = get_fidelities(result.final_density_matrix, idx)
        total_fab += fab
        total_fae += fae
    return total_fab / num_states, total_fae / num_states
def build_pccm():
    theta=np.pi/2
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

        ansatz.append(cirq.rx(symbols[12])(eve))
        ansatz.append(cirq.ry(symbols[13])(eve))
        ansatz.append(cirq.rz(symbols[14])(eve))
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
    initial_weights = np.zeros(18)
    result = minimize(qcl_loss, initial_weights, args=(noise), method='COBYLA',options={'maxiter': 100})

    resolver = dict(zip(symbols, result.x))
    resolved_circuit = cirq.resolve_parameters(ansatz_circuit, resolver)
    return evaluate_circuit(resolved_circuit, noise)

noise_levels = np.linspace(0.0, 0.2, 10) 
pccmfid = []
qclfid = []

pccm_circuit = build_pccm()


for step, i in enumerate(noise_levels):
    print(noise_levels)
    pccm_fid = evaluate_circuit(pccm_circuit, i)
    pccmfid.append(pccm_fid[1])
    qcl_fid = train_qcl(i)
    qclfid.append(qcl_fid[1])
    print("PCCM Eve Fid: " + str(pccm_fid[1]) + " QCL Eve Fid: " + str(qcl_fid[1]))
