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




circuit = cirq.Circuit()


import numpy as np



pccm = cirq.Circuit()
theta1 = np.cos(1/np.sqrt(2))

theta2 = np.sin(1/np.sqrt(2))

theta3 = -(np.cos(1/np.sqrt(2)))


pccm.append(cirq.ry(theta1*2)(eve))

pccm.append(cirq.CNOT(eve,ancilla))

pccm.append(cirq.ry(theta2*2)(ancilla))

pccm.append(cirq.CNOT(ancilla,eve))

pccm.append(cirq.ry(theta3*2)(eve))

pccm.append(cirq.CNOT(eve,signal))

pccm.append(cirq.CNOT(ancilla,signal))

pccm.append(cirq.CNOT(signal,eve))

pccm.append(cirq.CNOT(signal,ancilla))




fid = cirq.Circuit()

fid += my_protocol.alice_send_0_H_circuit
fid += pccm

sim = cirq.Simulator()

result = sim.simulate(fid)
state = result.final_state_vector
state=np.abs(state)**2
state
