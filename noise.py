#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cirq
import matplotlib.pyplot as plt


# In[2]:


signal = cirq.NamedQubit("signal")
eve = cirq.NamedQubit("eve")

simulator = cirq.DensityMatrixSimulator(seed = 42)

bb84_cases = [
    {
        "label": "0",
        "bit": 0,
        "basis": "Z",
        "preparation": cirq.Circuit(cirq.I(signal)),
        "target": np.array([1.0, 0.0], dtype=complex),
    },

    {
        "label": "1",
        "bit": 1,
        "basis": "Z",
        "preparation": cirq.Circuit(cirq.X(signal)),
        "target": np.array([0.0, 1.0], dtype=complex),
    },

    {
        "label": "+",
        "bit": 0,
        "basis": "X",
        "preparation": cirq.Circuit(cirq.H(signal)),
        "target": np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0),
    },

    {
        "label": "-",
        "bit": 1,
        "basis": "X",
        "preparation": cirq.Circuit(cirq.X(signal), cirq.H(signal)),
        "target": np.array([1.0, -1.0], dtype=complex) / np.sqrt(2.0),
    }
]

print("loaded", len(bb84_cases), "BB84 cases")


# In[3]:


num_params = 18

def build_interaction_u(num_params):
    """eve two-layer interaction u(theta) using params 0 - 11"""
    params = np.asarray(num_params, dtype=float)
    circuit = cirq.Circuit()

    for layer in range(2):
        offset = 6 * layer

        circuit.append(cirq.rx(params[offset + 0])(signal))
        circuit.append(cirq.ry(params[offset + 1])(signal))
        circuit.append(cirq.rz(params[offset + 2])(signal))

        circuit.append(cirq.rx(params[offset + 3])(eve))
        circuit.append(cirq.ry(params[offset + 4])(eve))
        circuit.append(cirq.rz(params[offset + 5])(eve))

        circuit.append(cirq.CNOT(signal, eve))

    return circuit


def build_measurement_v(num_params, basis):
    """eve basis-dependent unitary v(lambda) using params 12 - 17"""
    params = np.asarray(num_params, dtype=float)

    if basis == "Z":
        offset = 12
    elif basis == "X":
        offset = 15
    else:
        raise ValueError(f"unsupported basis {basis}")

    return cirq.Circuit(
        cirq.rx(params[offset + 0])(eve),
        cirq.ry(params[offset + 1])(eve),
        cirq.rz(params[offset + 2])(eve),
    )


test_params = np.zeros(num_params)

print("U(Theta):")
print(build_interaction_u(test_params))

print("\nV_Z(Lambda):")
print(build_measurement_v(test_params, "Z"))


# In[4]:


def build_attack_circuit(params, case):
    circuit = cirq.Circuit()
    circuit += case["preparation"]
    circuit.append(cirq.bit_flip(p=0.25).on(signal))

    circuit += build_interaction_u(params)
    circuit += build_measurement_v(params, case["basis"])

    return circuit


# In[5]:


# density matrices/fidelities
def reduced_density_matrices(joint_rho):
    rho_tensor = np.asarray(joint_rho).reshape(2, 2, 2, 2)

    bob_rho = np.trace(rho_tensor, axis1=1, axis2=3)
    eve_rho = np.trace(rho_tensor, axis1=0, axis2=2)

    return bob_rho, eve_rho


def raw_state_fidelity(rho, target_state):
    target_state = np.asarray(target_state, dtype=complex)
    value = np.vdot(target_state, rho @ target_state)

    return float(np.clip(np.real(value), 0.0, 1.0))

# simulate
def simulate_case(params, case):
    circuit = build_attack_circuit(params, case)

    result = simulator.simulate(circuit, qubit_order=[signal, eve])

    joint_rho = result.final_density_matrix
    bob_rho, eve_rho = reduced_density_matrices(joint_rho)

    return circuit, joint_rho, bob_rho, eve_rho

def average_fidelities(params, return_details = False):
    bob_scores = []
    eve_scores = []
    details = {}

    for case in bb84_cases:
        _, _, bob_rho, eve_rho = simulate_case(params,case)

        bob_score = raw_state_fidelity(bob_rho, case["target"])

        eve_score = float(
            np.clip(np.real(eve_rho[case["bit"], case["bit"]]), 0.0, 1.0)
        )

        bob_scores.append(bob_score)
        eve_scores.append(eve_score)

        details[case["label"]] = {
            "basis": case["basis"],
            "bit": case["bit"],
            "bob": bob_score,
            "eve": eve_score,
        }

    fab = float(np.mean(bob_scores))
    fae = float(np.mean(eve_scores))

    if return_details:
        return fab, fae, details

    return fab, fae


rng = np.random.default_rng(42)
initial_params = rng.normal(0.0, 0.1, size = num_params)

initial_fab, initial_fae, initial_details = average_fidelities(initial_params, return_details = True)

print(f"initial F_AB: {initial_fab:.6f}")
print(f"initial F_AE: {initial_fae:.6f}")
print(initial_details)


# In[6]:


# loss
def loss_value(fab, fae, target_bob, alpha = 10.0):
    return float(alpha * (fab - target_bob) ** 2 - fae)

def loss_and_gradient(params, target_bob, alpha = 10.0):
    params = np.asarray(params, dtype=float)

    fab, fae = average_fidelities(params)

    grad_fab = np.zeros_like(params)
    grad_fae = np.zeros_like(params)

    shift = np.pi / 2.0

    for index in range(len(params)):
        plus = params.copy()
        minus = params.copy()

        plus[index] += shift
        minus[index] -= shift

        fab_plus, fae_plus = average_fidelities(plus)
        fab_minus, fae_minus = average_fidelities(minus)

        grad_fab[index] = 0.5 * (fab_plus - fab_minus)
        grad_fae[index] = 0.5 * (fae_plus - fae_minus)

    loss = loss_value(fab, fae, target_bob, alpha)

    gradient = (2.0 * alpha * (fab - target_bob) * grad_fab - grad_fae)

    return loss, fab, fae, gradient


# In[7]:


# optimize adam
def train_qcl(target_bob, *, alpha = 10.0, learning_rate = 0.1, steps = 100, seed = 42, initial_scale = 0.1, verbose=True):
    rng = np.random.default_rng(seed)
    params = rng.normal(0.0, initial_scale, size = num_params)

    first_moment = np.zeros_like(params)
    second_moment = np.zeros_like(params)

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    history = {
        "step": [],
        "loss": [],
        "fab": [],
        "fae": [],
    }

    for step in range(1, steps + 1):
        loss, fab, fae, gradient = loss_and_gradient(params, target_bob, alpha = alpha)

        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * gradient**2

        first_unbiased = first_moment / (1.0 - beta1**step)
        second_unbiased = second_moment / (1.0 - beta2**step)

        params -= (learning_rate * first_unbiased / (np.sqrt(second_unbiased) + epsilon))

        params = (params + np.pi) % (2.0 * np.pi) - np.pi

        history["step"].append(step)
        history["loss"].append(loss)
        history["fab"].append(fab)
        history["fae"].append(fae)

        if verbose and (step == 1 or step % 10 == 0 or step == steps):
            print(f"step {step:3d} | "
                f"loss={loss: .6f} | "
                f"F_AB={fab:.6f} | "
                f"F_AE={fae:.6f}"
            )

    final_fab, final_fae, details = average_fidelities(params, return_details = True)

    return {
        "params": params,
        "fab": final_fab,
        "fae": final_fae,
        "details": details,
        "history": history,
        "target_bob": target_bob
    }
# In[8]:


symmetric_fidelity = 0.5 * (1.0 + 1.0 / np.sqrt(2.0))

target = 0.90

symmetric_result = train_qcl(target_bob = target, alpha = 10.0, learning_rate = 0.1, steps = 100, seed = 105)

print("\nSymmetric-point demonstration")
print(f"Training target f: {target:.6f}")
print(f"Expected PCCM point: ({symmetric_fidelity:.6f}, "
      f"{symmetric_fidelity:.6f})")
print(f"Final F_AB: {symmetric_result['fab']:.6f}")
print(f"Final F_AE: {symmetric_result['fae']:.6f}")
print(symmetric_result["details"])


# In[9]:


history = symmetric_result["history"]

fidelity_plot, loss_plot = plot_training(history, symmetric_fidelity = symmetric_fidelity, output_dir = "plots")

print("Saved:", fidelity_plot)
print("Saved:", loss_plot)


# In[10]:


target_values = np.array([0.55, 0.62, 0.70, 0.78, 0.853553, 0.90, 0.95, 0.98])

pareto_steps = 100

pareto_results = []

for run_index, target in enumerate(target_values):
    print(f"\nTraining target {run_index + 1}/{len(target_values)}: {target:.6f}")

    result = train_qcl(target_bob = float(target), alpha = 10.0, learning_rate = 0.1, steps = pareto_steps, seed = 100 + run_index, verbose = False)

    pareto_results.append(result)

    print(f"final F_AB={result['fab']:.6f}, "
        f"final F_AE={result['fae']:.6f}")


# In[11]:


pareto_plot = plot_pareto(pareto_results, symmetric_fidelity = symmetric_fidelity, output_dir="plots")

print("Saved:", pareto_plot)


# In[12]:


#pccm baseline
def build_known_pccm(theta, basis):
    circuit = cirq.Circuit()

    circuit.append(cirq.rx(np.pi / 2.0)(signal))
    circuit.append(cirq.ry(theta)(eve).controlled_by(signal))
    circuit.append(cirq.ry(-np.pi)(signal).controlled_by(eve))
    circuit.append(cirq.rx(-np.pi / 2.0)(signal))
    circuit.append(cirq.rx(-np.pi / 2.0)(eve))

    if basis == "X":
        circuit.append(cirq.H(eve))

    return circuit


def evaluate_known_pccm(theta):
    bob_scores = []
    eve_scores = []

    for case in bb84_cases:
        circuit = cirq.Circuit()
        circuit += case["preparation"]
        circuit.append(cirq.bit_flip(p=0.25).on(signal))

        circuit += build_known_pccm(theta, case["basis"])

        result = simulator.simulate(circuit, qubit_order=[signal, eve])

        bob_rho, eve_rho = reduced_density_matrices(result.final_density_matrix)

        bob_scores.append(raw_state_fidelity(bob_rho, case["target"]))
        eve_scores.append(float(np.clip(np.real(eve_rho[case["bit"], case["bit"]]), 0.0, 1.0)))

    return float(np.mean(bob_scores)), float(np.mean(eve_scores))


pccm_check = evaluate_known_pccm(np.pi / 2.0)

print("Known PCCM at theta=pi/2:")
print(f"F_AB={pccm_check[0]:.6f}")
print(f"F_AE={pccm_check[1]:.6f}")


# In[ ]:
