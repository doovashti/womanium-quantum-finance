# src/problem_demo.py
# Minimal binary QP demo with linear constraints (Task 1)
# -------------------------------------------------------
# We create a small 0/1 selection problem:
#   minimize    x^T Q x + q^T x + c0
#   subject to  A_le x <= b_le
#               x in {0,1}^n
#
# Intuition (portfolio/knapsack flavor):
#   - Quadratic term (risk):   lambda_risk * x^T Sigma x  (penalizes picking correlated items)
#   - Linear term (reward):   -mu^T x                     (encourages high-return items)
#   - Constraints:
#         1) sum(cost_i * x_i) <= budget
#         2) sum(x_i) <= k  (cardinality cap)
#
# This is small, runs fast, and is easy to explain in the write-up.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
def build_demo_problem(n: int = 8):
    # (this matches the prints you saw earlier)
    rng = np.random.default_rng(7)
    # symmetric PSD-ish Q for demo
    M = rng.normal(0, 0.1, size=(n, n))
    Q = (M + M.T) / 2
    # linear term
    q = rng.normal(-0.12, 0.05, size=n)

    # two linear <= constraints: A x <= b
    A_le = np.array([
        [1, 3, 1, 2, 3, 1, 2, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype=float)
    b_le = np.array([6.0, 3.0], dtype=float)

    return Q, q, A_le, b_le

if __name__ == "__main__":
    Q, q, A_le, b_le = build_demo_problem(8)
    print("n:", Q.shape[0])
    print("Q shape:", Q.shape, " (symmetric, first 3x3):\n", np.round(Q[:3,:3], 3))
    print("q (first 5):", np.round(q[:5], 3))
    print("A_le:\n", A_le)
    print("b_le:", b_le)

@dataclass
class BinaryQP:
    n: int                      # number of binary variables
    Q: np.ndarray               # (n,n) symmetric matrix for quadratic term
    q: np.ndarray               # (n,) vector for linear term
    c0: float                   # scalar constant
    A_le: Optional[np.ndarray]  # (m,n) for <= constraints (or None)
    b_le: Optional[np.ndarray]  # (m,) for <= constraints (or None)
    var_names: List[str]        # names for readability

def _random_spd(n: int, rng: np.random.Generator) -> np.ndarray:
    """Create a small symmetric positive-definite matrix."""
    A = rng.normal(size=(n, n))
    Sigma = A @ A.T  # PSD
    # add small diagonal shift to make it better conditioned
    Sigma += 0.1 * np.eye(n)
    # normalize scale so numbers aren't huge
    Sigma /= np.max(np.abs(Sigma))
    return Sigma

def demo_portfolio_small(
    n: int = 8,
    k: int = 3,
    budget: int = 6,
    lambda_risk: float = 0.2,
    seed: int = 7,
) -> BinaryQP:
    """
    Build a tiny binary QP with:
      minimize   lambda_risk * x^T Sigma x  - mu^T x
      s.t.       cost^T x <= budget
                 sum(x)    <= k
                 x in {0,1}^n

    Returned in canonical form:  x^T Q x + q^T x + c0
    """
    rng = np.random.default_rng(seed)

    # synthetic "returns" (mu) and "costs"
    mu = rng.uniform(0.05, 0.20, size=n)             # larger is better
    cost = rng.integers(low=1, high=4, size=n)       # small integer costs 1..3

    # correlated "risk" matrix
    Sigma = _random_spd(n, rng)

    # Build objective in the form x^T Q x + q^T x + c0
    # risk part (minimize) + reward part (as negative linear term)
    Q = lambda_risk * Sigma
    # NOTE: We do not need the 1/2 factor; we can fold it into Q consistently elsewhere.
    q = -mu
    c0 = 0.0

    # Linear constraints (<=)
    # 1) budget
    A1 = cost.reshape(1, n)
    b1 = np.array([budget], dtype=float)

    # 2) cardinality <= k
    A2 = np.ones((1, n))
    b2 = np.array([float(k)])

    A_le = np.vstack([A1, A2])
    b_le = np.concatenate([b1, b2])

    var_names = [f"x_{i}" for i in range(n)]

    return BinaryQP(
        n=n,
        Q=Q.astype(float),
        q=q.astype(float),
        c0=float(c0),
        A_le=A_le.astype(float),
        b_le=b_le.astype(float),
        var_names=var_names,
    )

# Convenience main for quick inspection
if __name__ == "__main__":
    prob = demo_portfolio_small()
    print("n:", prob.n)
    print("Q shape:", prob.Q.shape, " (symmetric, first 3x3):\n", np.round(prob.Q[:3,:3], 3))
    print("q (first 5):", np.round(prob.q[:5], 3))
    print("A_le:\n", prob.A_le)
    print("b_le:", prob.b_le)
