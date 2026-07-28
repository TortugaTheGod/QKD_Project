#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cirq
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from bb84 import BB84


# In[3]:


my_protocol = BB84(eve_intercept="yes")


# In[4]:


signal = cirq.NamedQubit("signal")
eve = cirq.NamedQubit("eve")
ancilla = cirq.NamedQubit("ancilla")

sim = cirq.DensityMatrixSimulator()

my_protocol.qubit = signal
my_protocol.simulator = sim


# In[5]:


# alice Circuit
my_protocol.alice_send_0_no_H_circuit = cirq.Circuit(cirq.I(signal))
my_protocol.alice_send_1_no_H_circuit = cirq.Circuit(cirq.X(signal))
my_protocol.alice_send_0_H_circuit = cirq.Circuit(cirq.H(signal))
my_protocol.alice_send_1_H_circuit = cirq.Circuit(cirq.X(signal), cirq.H(signal))

prep_circuits = {
    "0": my_protocol.alice_send_0_no_H_circuit,
    "1": my_protocol.alice_send_1_no_H_circuit,
    "+": my_protocol.alice_send_0_H_circuit,
    "-": my_protocol.alice_send_1_H_circuit,
}

target_states = {
    "0": np.array([1.0, 0.0], dtype=complex),
    "1": np.array([0.0, 1.0], dtype=complex),
    "+": np.array([1.0, 1.0], dtype=complex) / np.sqrt(2),
    "-": np.array([1.0, -1.0], dtype=complex) / np.sqrt(2),
}


# In[6]:


# pccm circuit
def build_pccm(theta):
    """Build two-qubit parameterized cloning ansatz."""
    theta = float(np.asarray(theta).reshape(-1)[0])

    pccm = cirq.Circuit()

    pccm.append(cirq.rx(np.pi / 2)(signal))
    pccm.append(cirq.ry(theta)(eve).controlled_by(signal))
    pccm.append(cirq.ry(-np.pi)(signal).controlled_by(eve))
    pccm.append(cirq.rx(-np.pi / 2)(signal))
    pccm.append(cirq.rx(-np.pi / 2)(eve))

    return pccm
print(build_pccm(0.0))


# In[7]:


# get eve/bob reduced states
def reduced_density_matrices(two_qubit_rho):
    """Trace out each qubit from 4x4 two-qubit density matrix."""
    rho_tensor = np.asarray(two_qubit_rho).reshape(2, 2, 2, 2)

    # signal_row, eve_row, signal_col, eve_col.
    bob_rho = np.trace(rho_tensor, axis1=1, axis2=3)
    eve_rho = np.trace(rho_tensor, axis1=0, axis2=2)

    return bob_rho, eve_rho

# attack simulation
def simulate_attack(theta, prep_circuit):
    full_circuit = cirq.Circuit()
    full_circuit += prep_circuit
    full_circuit += build_pccm(theta)

    result = sim.simulate(full_circuit, qubit_order=[signal, eve])

    joint_rho = result.final_density_matrix
    bob_rho, eve_rho = reduced_density_matrices(joint_rho)

    return full_circuit, joint_rho, bob_rho, eve_rho


# In[8]:


# calc fidelities
def raw_state_fidelity(density_matrix, target_state):
    target_state = np.asarray(target_state, dtype = complex)
    value = np.vdot(target_state, density_matrix @ target_state)

    return float(np.clip(np.real(value), 0.0, 1.0))


def eval_theta(theta):
    per_state = {}

    for label, prep_circuit in prep_circuits.items():
        _, _, bob_rho, eve_rho = simulate_attack(theta, prep_circuit)
        target = target_states[label]

        per_state[label] = {
            "bob": raw_state_fidelity(bob_rho, target),
            "eve": raw_state_fidelity(eve_rho, target),
        }

    bob_average = float(np.mean([values["bob"] for values in per_state.values()]))
    eve_average = float(np.mean([values["eve"] for values in per_state.values()]))

    return bob_average, eve_average, per_state


bob_fidelity, eve_fidelity, details = eval_theta(0.0)

print(f"Bob average fidelity @ theta = 0: {bob_fidelity:.6f}")
print(f"Eve average fidelity @ theta = 0: {eve_fidelity:.6f}")
print(details)


# In[9]:


fidelity = 0.5 * (1.0 + 1.0 / np.sqrt(2.0))
symmetry_weight = 0.25

def loss_function(theta):
    bob_fidelity, eve_fidelity, _ = eval_theta(theta)

    target_error = ((bob_fidelity - fidelity) ** 2 + (eve_fidelity - fidelity) ** 2)
    symmetry_error = symmetry_weight * (bob_fidelity - eve_fidelity) ** 2

    return float(target_error + symmetry_error)

print(f"Target PCCM fidelity: {fidelity:.6f}")
print(f"Loss at theta = 0: {loss_function([0.0]):.6f}")


# In[10]:


#optimize theta
optimizations = []

for initial_theta in np.linspace(-np.pi, np.pi, 9):
    run = minimize(loss_function, x0 = np.array([initial_theta]), method = "Nelder-Mead", options = {"maxiter": 500, "xatol": 1e-10, "fatol": 1e-12})
    optimizations.append(run)

best_result = min(optimizations, key=lambda run: run.fun)
best_theta = float(best_result.x[0])

best_bob_fidelity, best_eve_fidelity, best_details = eval_theta(best_theta)

print("Optimization success:", best_result.success)
print("Optimizer message:", best_result.message)

print(f"Best theta: {best_theta:.10f} radians")
print(f"Best loss: {best_result.fun:.10f}")
print(f"Bob avg fidelity: {best_bob_fidelity:.6f}")
print(f"Eve avg fidelity: {best_eve_fidelity:.6f}")

print("Per-state fidelities:", best_details)


# In[11]:


# plots for data
theta_vals = np.linspace(-np.pi, np.pi, 241)
bob_vals = []
eve_vals = []
loss_vals = []

for value in theta_vals:
    bob_value, eve_value, _ = eval_theta(value)
    bob_vals.append(bob_value)
    eve_vals.append(eve_value)
    loss_vals.append(loss_function([value]))

plt.figure(figsize=(9, 5))
plt.plot(theta_vals, bob_vals, label="Bob average fidelity")
plt.plot(theta_vals, eve_vals, label="Eve average fidelity")
plt.axhline(fidelity, linestyle="--", label="PCCM target")
plt.axvline(best_theta, linestyle=":", label="Optimized theta")
plt.xlabel("theta (radians)")
plt.ylabel("Average fidelity")
plt.title("QCL fidelity landscape")
plt.legend()
plt.grid(alpha=0.25)
plt.show()

plt.figure(figsize=(9, 5))
plt.plot(theta_vals, loss_vals)
plt.axvline(best_theta, linestyle=":", label="Optimized theta")
plt.xlabel("theta (radians)")
plt.ylabel("Loss")
plt.title("QCL loss landscape")
plt.legend()
plt.grid(alpha=0.25)
plt.show()


# In[12]:


# final cirquit
optimized_circuit = build_pccm(best_theta)

print("Optimized PCCM ansatz:")
print(optimized_circuit)


# In[ ]:




