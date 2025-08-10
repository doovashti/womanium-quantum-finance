# src/compare.py
import json, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .problem_demo import build_demo_problem
from .qubo import penalize_le_constraints_to_qubo

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def objective_and_violations(Q, q, A, b, z):
    z = np.asarray(z, dtype=float)
    f = float(z @ Q @ z + q @ z)
    viol = np.maximum(A @ z - b, 0.0)
    return f, viol

def brute_force_best(Q, q, A, b):
    n = Q.shape[0]
    best_f = float("inf")
    best_z = None
    for x in range(1 << n):
        z = np.array([(x >> i) & 1 for i in range(n)], dtype=float)
        f, viol = objective_and_violations(Q, q, A, b, z)
        if np.all(viol == 0) and f < best_f:
            best_f, best_z, best_viol = f, z, viol
    return best_f, best_z, best_viol

def simulated_annealing(Q, q, A, b, T0=2.0, cooling=0.995, steps=20_000, seed=7):
    rng = np.random.default_rng(seed)
    n = Q.shape[0]

    # start with a random bitstring
    z = rng.integers(0, 2, size=n, dtype=int)
    f = float(z @ Q @ z + q @ z)

    # constraint satisfaction flags and helper to make a weighted lexicographic score
    def sat_flags(zv):
        ok = (A @ zv <= b + 1e-9)           # booleans
        return ok.astype(int)               # cast to ints (0/1)

    def lex_score(flags: np.ndarray) -> int:
        # larger is better; weight constraints in priority order
        # (example: two constraints → weight[0]=1e6, weight[1]=1e3)
        # generalize weights if you add constraints
        weights = np.array([1_000_000, 1_000], dtype=int)[: len(flags)]
        return int(np.dot(weights, flags))

    f_best, z_best = f, z.copy()
    s_cur = sat_flags(z)
    s_best = s_cur.copy()

    T = T0
    for _ in range(steps):
        # propose a 1-bit flip
        i = rng.integers(0, n)
        z_new = z.copy()
        z_new[i] ^= 1  # flip

        f_new = float(z_new @ Q @ z_new + q @ z_new)
        s_new = sat_flags(z_new)

        # compute "energy" with lexicographic preference to constraint satisfaction
        # accept if: (1) more constraints satisfied, or (2) same constraints but better objective,
        # otherwise accept with Boltzmann probability
        score_cur = lex_score(s_cur)
        score_new = lex_score(s_new)

        better = (score_new > score_cur) or (score_new == score_cur and f_new < f)
        if better:
            accept = True
        else:
            # temperature-based acceptance on a combined delta
            # penalize objective more when constraints are worse
            delta_score = score_new - score_cur
            # scale lexicographic score so it dominates objective at high magnitude
            combined_delta = -delta_score * 1e3 + (f_new - f)
            accept = rng.random() < np.exp(-combined_delta / max(T, 1e-12))

        if accept:
            z, f, s_cur = z_new, f_new, s_new
            if (score_new > lex_score(s_best)) or (score_new == lex_score(s_best) and f_new < f_best):
                z_best, f_best, s_best = z.copy(), f_new, s_new.copy()

        T *= cooling

    # return objective (original), bitstring, and constraint violations
    violations = np.maximum(A @ z_best - b, 0.0)
    return f_best, z_best, violations

def load_quantum_solution(n):
    """Try to load the bitstring produced by vqe_solve.py; return None if missing."""
    out_file = RESULTS_DIR / "vqe_solution.json"
    if not out_file.exists():
        return None
    data = json.loads(out_file.read_text())
    z_all = np.array(data.get("bitstring", []), dtype=int)
    if z_all.size < n:
        return None
    return z_all[:n]

def main():
    n = 8
    Q, q, A, b = build_demo_problem(n)

    # quantum candidate (from saved VQE run)
    z_quantum = load_quantum_solution(n)
    quantum_info = {}
    if z_quantum is None:
        quantum_info = {"note": "No vqe_solution.json found in results/. Run `python -m src.vqe_solve` first."}
    else:
        fQ, vQ = objective_and_violations(Q, q, A, b, z_quantum)
        # “energy” is the penalized QUBO objective value at the lifted vector (for reference)
        Qq, qq, c = penalize_le_constraints_to_qubo(Q, q, A, b, penalty=1000.0)
        zQ_full = np.concatenate([z_quantum, np.zeros(Qq.shape[0]-n)])  # slack bits set to 0 for reference
        energy = float(zQ_full @ Qq @ zQ_full + qq @ zQ_full + c)
        quantum_info = {"f": fQ, "violations": vQ.tolist(), "energy": energy, "z": z_quantum.astype(int).tolist()}

    # exact (feasible optimum)
    f_exact, z_exact, v_exact = brute_force_best(Q, q, A, b)

    # simulated annealing
    f_sa, z_sa, v_sa = simulated_annealing(Q, q, A, b)

    # print
    print("\n=== Comparison ===")
    if z_quantum is None:
        print("Quantum: (no saved solution yet)")
    else:
        print(f"Quantum: f={quantum_info['f']:.6f}, violations={list(np.array(quantum_info['violations']))}, energy={quantum_info['energy']:.3f}")
    print(f"Exact  : f={f_exact:.6f}, violations={list(v_exact)}")
    print(f"SA     : f={f_sa:.6f}, violations={list(v_sa)}")

    # save JSON
    out = {
        "quantum": quantum_info,
        "exact": {"f": f_exact, "violations": v_exact.tolist(), "z": z_exact.astype(int).tolist()},
        "sa": {"f": f_sa, "violations": v_sa.tolist(), "z": z_sa.astype(int).tolist()},
    }
    (RESULTS_DIR / "comparison.json").write_text(json.dumps(out, indent=2))

    # bar chart for presentation
    labels = ["Quantum", "SimAnneal", "Exact"]
    vals = [
        quantum_info["f"] if z_quantum is not None else np.nan,
        f_sa,
        f_exact,
    ]
    plt.figure()
    plt.bar(labels, vals)
    plt.ylabel("Objective (lower is better)")
    plt.title("Objective comparison")
    plt.savefig(RESULTS_DIR / "comparison.png", bbox_inches="tight")
    # no plt.show() so it doesn't block in CI/terminal

if __name__ == "__main__":
    main()
