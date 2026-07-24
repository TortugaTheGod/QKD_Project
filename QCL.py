my_protocol = BB84(eve_intercept='yes')



my_qubits = cirq.NamedQubit.range(3, prefix='q')

signal = my_qubits[0]
eve = my_qubits[1]
ancilla = my_qubits[2]
my_protocol.qubit = signal
my_protocol.simulator = cirq.Simulator()



# alice Circuit
circuit = cirq.Circuit()

circuit.append(cirq.I(signal))

my_protocol.alice_send_0_no_H_circuit = circuit



circuit = cirq.Circuit()

circuit.append(cirq.I(signal))
circuit.append(cirq.H(signal))

my_protocol.alice_send_0_H_circuit = circuit


# bob circ
circuit = cirq.Circuit()


circuit.append(cirq.I(signal))
circuit.append(cirq.measure(signal))

my_protocol.bob_receive_no_H_circuit = circuit


sim = cirq.Simulator()

circuit = cirq.Circuit()


import numpy as np


theta = 0


def build_pccm(theta, alice_used_H=False):
  pccm = cirq.Circuit()

  pccm.append(cirq.rx(np.pi/2)(signal))
  pccm.append(cirq.ry(theta)(eve).controlled_by(signal))
  pccm.append(cirq.ry(-(np.pi))(signal).controlled_by(eve))
  pccm.append(cirq.rx(-(np.pi/2))(signal))
  pccm.append(cirq.rx(-(np.pi/2))(eve))
  return pccm

 # if bob and alice use H, append H to eve's bit in pccm
if my_protocol.does_alice_apply_H == "yes" and my_protocol.does_bob_apply_H == "yes":
    pccm.append(cirq.H(eve))

def loss_function(theta):
    pccm = build_pccm(theta, alice_used_H = True)

    circuit = cirq.Circuit()

    circuit += my_protocol.alice_send_0_H_circuit
    circuit += pccm

    result = sim.simulate(circuit)

    fidelity = ???????
    return fidelity

circuit = cirq.Circuit()

fid = cirq.Circuit()

fid += my_protocol.alice_send_0_H_circuit
fid += pccm
"""
Pure Fidelity (if you have state vectors)
psi = psi / np.linalg.norm(psi)
phi = phi / np.linalg.norm(phi)
fidelity = float(np.abs(np.vdot(psi, phi)) ** 2)

Density Fidelity (if you have density matrices)
given matrix rho_a and rho_b, 
fidelity = np.trace(np.sqrt(np.sqrt(rho_a) * rho_b * np.sqrt(rho_a)))**2
"""
result = sim.simulate(fid)
state = result.final_state_vector
state=np.abs(state)**2
state
