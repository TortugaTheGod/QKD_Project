from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

def _prepare_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents = True, exist_ok = True)
    
    return path


def plot_training(history: dict[str, list[float]], symmetric_fidelity: float, output_dir: str | Path = "plots", show: bool = True) -> tuple[Path, Path]:
    """plot & save training fidelities and loss"""
    output_path = _prepare_output_dir(output_dir)

    fidelity_path = output_path / "qcl_training_fidelities.png"
    loss_path = output_path / "qcl_training_loss.png"

    fig = plt.figure(figsize=(9, 5))
    
    plt.plot(history["step"], history["fab"], label="F_AB")
    plt.plot(history["step"], history["fae"], label="F_AE")
    
    plt.axhline(symmetric_fidelity, linestyle = "--", label = "Symmetric PCCM fidelity")
    
    plt.xlabel("training step")
    plt.ylabel("avg fidelity")
    plt.title("QCL training convergence")
    plt.legend()
    plt.grid(alpha = 0.25)
    plt.tight_layout()
    fig.savefig(fidelity_path, dpi=200, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)

    fig = plt.figure(figsize = (9, 5))
    plt.plot(history["step"], history["loss"])
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.title("QCL loss during training")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(loss_path, dpi = 200, bbox_inches = "tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fidelity_path, loss_path


def plot_pareto(pareto_results: list[dict[str, Any]], symmetric_fidelity: float, output_dir: str | Path = "plots", show: bool = True) -> Path:
    output_path = _prepare_output_dir(output_dir)
    pareto_path = output_path / "qcl_pareto_frontier.png"

    trained_fab = np.array([result["fab"] for result in pareto_results])
    trained_fae = np.array([result["fae"] for result in pareto_results])

    theta_values = np.linspace(0.0, np.pi, 400)
    analytic_fab = (1.0 + np.cos(theta_values / 2.0)) / 2.0
    analytic_fae = (1.0 + np.sin(theta_values / 2.0)) / 2.0

    fig = plt.figure(figsize=(8, 6))
    plt.plot(analytic_fab, analytic_fae, label="Analytic PCCM frontier")
    plt.scatter(trained_fab, trained_fae, s = 55, label="QCL final results")
    plt.scatter([symmetric_fidelity], [symmetric_fidelity], marker = "x", s = 100, label="Symmetric PCCM point")
    plt.xlabel(r"Alice Bob fidelity $F_{AB}$")
    plt.ylabel(r"Alice Eve fidelity $F_{AE}$")
    plt.title("BB84 individual-attack Pareto frontier")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(pareto_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return pareto_path
