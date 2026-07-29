import cirq
import sympy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

signal = cirq.NamedQubit('signal')
eve = cirq.NamedQubit('eve')
sim = cirq.DensityMatrixSimulator(dtype=np.complex128)

target_state = [np.array([1, 0]), np.array([0, 1]),np.array([1/np.sqrt(2), 1/np.sqrt(2)]),np.array([1/np.sqrt(2), -1/np.sqrt(2)])]

def get_fidelities(rho_4x4, idx):
    rho = rho_4x4.reshape((2, 2, 2, 2))
    bob_rho = np.trace(rho, axis1=1, axis2=3)
    eve_rho = np.trace(rho, axis1=0, axis2=2)
    target_vec = target_state[idx]
    fab = np.real(target_vec.conj() @ bob_rho @ target_vec)
    fae = np.real(target_vec.conj() @ eve_rho @ target_vec)
    return fab, fae

def evaluate_circuit(circ):
    total_fab, total_fae = 0.0, 0.0
    num_states = len(target_state)
    for idx in range(num_states):
        circuit = cirq.Circuit()
        if idx == 0:
            circuit.append(cirq.I(signal))
        elif idx == 1:
            circuit.append(cirq.X(signal))
        elif idx == 2:
            circuit.append(cirq.H(signal))
        elif idx == 3:
            circuit.append(cirq.X(signal))
            circuit.append(cirq.H(signal))

        circuit.append(circ)
        result = sim.simulate(circuit, qubit_order=[signal, eve])
        fab, fae = get_fidelities(result.final_density_matrix, idx)
        total_fab += fab
        total_fae += fae
    return total_fab / num_states, total_fae / num_states

def build_pccm(theta):
    pccm = cirq.Circuit()
    pccm.append(cirq.rx(np.pi/2)(signal))
    pccm.append(cirq.ry(theta)(eve).controlled_by(signal))
    pccm.append(cirq.ry(-np.pi)(signal).controlled_by(eve))
    pccm.append(cirq.rx(-np.pi/2)(signal))
    pccm.append(cirq.rx(-np.pi/2)(eve))
    return pccm

thetas = np.linspace(0, np.pi, 100)
pccm_fab = []
pccm_fae = []

for t in thetas:
    pccm_circ = build_pccm(t)
    fab, fae = evaluate_circuit(pccm_circ)
    pccm_fab.append(fab)
    pccm_fae.append(fae)

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
    ansatz.append(cirq.rx(symbols[15])(eve))
    ansatz.append(cirq.ry(symbols[16])(eve))
    ansatz.append(cirq.rz(symbols[17])(eve))
    return ansatz

ansatz_circuit = build_qcl_ansatz()


target_fidelities = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
alpha = 25

qcl_runs = []

qcl_fab = []
qcl_fae = []

for f_target in target_fidelities:
    history = []

    def qcl_loss(weights):
        resolver = dict(zip(symbols, weights))
        resolved_circuit = cirq.resolve_parameters(ansatz_circuit, resolver)
        fab, fae = evaluate_circuit(resolved_circuit)
        history.append((fab, fae))
        loss = alpha * ((fab - f_target) ** 2) - fae
        return loss

    initial_weights = np.random.normal(0, np.pi, 18)
    minimize(qcl_loss, initial_weights, method='COBYLA', options={'maxiter': 500})
    
    final_fab, final_fae = history[-1]
    qcl_fab.append(final_fab)
    qcl_fae.append(final_fae)


qcl_data = sorted(zip(qcl_fab, qcl_fae), key=lambda x: x[0])
qcl_fab_sorted = [x[0] for x in qcl_data]
qcl_fae_sorted = [x[1] for x in qcl_data]


plt.figure(figsize=(7, 5))

plt.plot(pccm_fab, pccm_fae, color='blue', linewidth=2, label='PCCM')
plt.plot(qcl_fab_sorted, qcl_fae_sorted, color='orange', linewidth=2, label='QCL')

plt.title("PCCM vs QCL", fontsize=12)
plt.xlabel("Alice Bob Fidelity ", fontsize=11)
plt.ylabel("Alice/Eve Fidelity ", fontsize=11)

plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()

plt.show()
