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


theta = 1 #NEEDS TO BE TRAINED 

pccm.append(cirq.rx(np.pi/2)(signal))

pccm.append(cirq.ry(theta)(eve).controlled_by(signal))
pccm.append(cirq.ry(-(np.pi/2))(signal).controlled_by(eve))

pccm.append(cirq.rx(-(np.pi/2))(signal))
pccm.append(cirq.rx(-(np.pi/2))(signal))

 # if bob and alice use H, append H to eve's bit in pccm

circuit = cirq.Circuit()

fid = cirq.Circuit()

fid += my_protocol.alice_send_0_H_circuit
fid += pccm

sim = cirq.Simulator()

result = sim.simulate(fid)
state = result.final_state_vector
state=np.abs(state)**2
state
