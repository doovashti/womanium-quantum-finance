# src/vqe_solve.py

import numpy as np
import json

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms import VQE
from qiskit.primitives import Estimator, Sampler  # V1 primitives (deprecated but fine here)
from pathlib import Path

# make sure you also import the QUBO helpers from your qubo.py
from .qubo import penalize_le_constraints_to_qubo, qubo_to_ising

from .problem_demo import build_demo_problem
from .qubo import penalize_le_constraints_to_qubo, qubo_to_ising


def ising_operator_from_h_J(h: np.ndarray, J: np.ndarray) -> SparsePauliOp:
    """
    Build a SparsePauliOp for Ising Hamiltonian:
        H = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j
    J is assumed symmetric; only upper triangle is used.
    """
    N = len(h)
    paulis = []
    coeffs = []

    # linear Z terms
    for i in range(N):
        z = ["I"] * N
        z[i] = "Z"
        paulis.append("".join(z[::-1]))  # Qiskit uses little-endian ordering (rightmost is qubit 0)
        coeffs.append(h[i])

    # quadratic ZZ terms
    for i in range(N):
        for j in range(i + 1, N):
            if J[i, j] != 0.0:
                z = ["I"] * N
                z[i] = "Z"
                z[j] = "Z"
                paulis.append("".join(z[::-1]))
                coeffs.append(J[i, j])

    return SparsePauliOp.from_list(list(zip(paulis, map(complex, coeffs))))

def qubo_to_pauli_op(Q: np.ndarray, q: np.ndarray, offset_qubo: float, ansatz_reps: int = 3):
    """
    Build an Ising Hamiltonian H from QUBO (Q, q, offset) and a simple TwoLocal ansatz.
    Returns: (H: SparsePauliOp, ansatz: QuantumCircuit)
    """
    # QUBO -> Ising (z in {-1,1}):
    h, J, const = qubo_to_ising(Q, q, offset_qubo)

    N = Q.shape[0]

    # Move any diagonal part of J into the constant (since z_i^2 = 1)
    const += float(np.trace(J))
    J = J.copy()
    np.fill_diagonal(J, 0.0)

    labels = []
    coeffs = []

    # constant term
    labels.append("I" * N)
    coeffs.append(const)

    # linear Z terms: label order in SparsePauliOp is qubit_{N-1} ... qubit_0
    for i in range(N):
        lab = "I" * (N - 1 - i) + "Z" + "I" * i
        labels.append(lab)
        coeffs.append(float(h[i]))

    # quadratic ZZ terms: use only i<j and sum symmetric entries (to avoid double count)
    for i in range(N):
        for j in range(i + 1, N):
            cij = float(J[i, j] + J[j, i])  # equals 2*J[i,j] if J is symmetric
            if cij != 0.0:
                lab_list = ["I"] * N
                lab_list[N - 1 - i] = "Z"
                lab_list[N - 1 - j] = "Z"
                labels.append("".join(lab_list))
                coeffs.append(cij)

    H = SparsePauliOp.from_list(list(zip(labels, coeffs)))

    # simple (but solid) ansatz
    ansatz = TwoLocal(
        N,
        rotation_blocks="ry",
        entanglement_blocks="cz",
        entanglement="full",
        reps=ansatz_reps,
    )
    return H, ansatz


def main():
    # --- 1) Build demo problem and QUBO/Ising ---
    from .problem_demo import build_demo_problem
    from .qubo import penalize_le_constraints_to_qubo, qubo_to_ising

    n = 8  # original decision bits
    Q, q, A_le, b_le = build_demo_problem(n)

    Q_qubo, q_qubo, offset = penalize_le_constraints_to_qubo(Q, q, A_le, b_le, penalty=1000.0)

    # Turn QUBO into Pauli H and an ansatz
    H, ansatz = qubo_to_pauli_op(Q_qubo, q_qubo, offset, ansatz_reps=2)
    N = ansatz.num_qubits  # total qubits incl. slack bits

    # --- 2) Manual VQE (statevector-based expectation + COBYLA) ---
    from qiskit.quantum_info import Statevector
    from scipy.optimize import minimize
    import numpy as np

    def energy(theta: np.ndarray) -> float:
        bound = ansatz.assign_parameters(theta, inplace=False)
        psi = Statevector.from_instruction(bound)
        # SparsePauliOp expectation_value returns complex; real part is the energy
        return float(np.real(psi.expectation_value(H)))

    theta0 = np.zeros(ansatz.num_parameters, dtype=float)
    opt = minimize(
    energy, theta0, method="COBYLA",
    options={"maxiter": 2000, "rhobeg": 0.5, "tol": 1e-3, "disp": True}
)
    
    history = []
    def cb(theta):
    # reuse the same energy(theta) function you already have
        val = energy(theta)
        history.append(val)

# then in minimize(...), pass: callback=cb
    opt = minimize(energy, theta0, method="COBYLA",
               options={"maxiter": 2000, "disp": True},
               callback=cb)

# after optimization:
    import csv, pathlib
    results_dir = pathlib.Path("results"); results_dir.mkdir(exist_ok=True)
    with open(results_dir/"vqe_convergence.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iter","energy"])
        for i, e in enumerate(history): w.writerow([i, e])

    # --- 3) Sample the optimized circuit to get a bitstring candidate ---
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    bound = ansatz.assign_parameters(opt.x, inplace=False)
    meas = bound.copy()
    meas.measure_all()

    backend = AerSimulator()
    tmeas = transpile(meas, backend)
    job = backend.run(tmeas, shots=2000)
    counts = job.result().get_counts()

    # most frequent bitstring; Qiskit returns little-endian, so reverse for MSB->LSB
    bitstr = max(counts, key=counts.get)
    z_all = np.array(list(map(int, bitstr[::-1])), dtype=int)  # length N

    # First n bits correspond to original decision variables (slacks are appended after)
    z = z_all[:n]

    # --- 4) Evaluate original objective + constraint violations on z ---
    obj_val = float(z @ Q @ z + q @ z)
    violations = np.maximum(A_le @ z - b_le, 0.0)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out = {
        "bitstring": list(map(int, z_all.tolist())),   # all bits (incl. slacks), little→big handled above
        "z_first_n": list(map(int, z.tolist())),       # first n decision bits
        "objective": float(obj_val),
        "violations": list(map(float, violations.tolist())),
        "vqe_energy": float(opt.fun),
        "n_params": int(ansatz.num_parameters),
    }
    (results_dir / "vqe_solution.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved VQE solution to {results_dir/'vqe_solution.json'}")

    print("VQE result:")
    print("  energy:", opt.fun)
    print("  n_params:", ansatz.num_parameters)
    iters  = getattr(opt, "nit", None)
    fevals = getattr(opt, "nfev", None)
    print("  iters:", iters if iters is not None else "(n/a)",
        "  function evals:", fevals if fevals is not None else "(n/a)",
        "  success:", opt.success)
    
def run_vqe_once(Q, q, A, b, penalty=1000.0, ansatz_reps=3, maxiter=500, seed=0):
    from .qubo import penalize_le_constraints_to_qubo
    from qiskit.quantum_info import Statevector
    from scipy.optimize import minimize
    import numpy as np

    Q_qubo, q_qubo, offset = penalize_le_constraints_to_qubo(Q, q, A, b, penalty=penalty)
    H, ansatz = qubo_to_pauli_op(Q_qubo, q_qubo, offset, ansatz_reps=ansatz_reps)

    rng = np.random.default_rng(seed)
    theta0 = 0.1 * rng.standard_normal(ansatz.num_parameters)

    def energy(theta):
        psi = Statevector.from_instruction(ansatz.assign_parameters(theta, inplace=False))
        return float(np.real(psi.expectation_value(H)))

    opt = minimize(energy, theta0, method="COBYLA",
                   options={"maxiter": maxiter, "disp": False})
    # sample most likely bitstring and return first n bits
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    meas = ansatz.assign_parameters(opt.x, inplace=False)
    meas.measure_all()
    backend = AerSimulator()
    counts = backend.run(transpile(meas, backend), shots=2000).result().get_counts()
    bitstr = max(counts, key=counts.get)
    z_all = np.array(list(map(int, bitstr[::-1])), dtype=int)
    return z_all[: len(q)], opt.fun


if __name__ == "__main__":
    main()
