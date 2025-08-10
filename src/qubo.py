# src/qubo.py

import numpy as np
from typing import Tuple


def _binary_slack_bits(upper_bound: int) -> int:
    """Smallest m such that 2^m - 1 >= upper_bound (encode [0..upper_bound])."""
    if upper_bound <= 0:
        return 0
    m = 0
    cap = 0
    while cap < upper_bound:
        m += 1
        cap = (1 << m) - 1
    return m


def penalize_le_constraints_to_qubo(
    Q: np.ndarray,
    q: np.ndarray,
    A_le: np.ndarray,
    b_le: np.ndarray,
    penalty: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Turn a binary quadratic problem with linear <= constraints into an unconstrained QUBO
        minimize x^T Q x + q^T x
        s.t.     A_le x <= b_le
    by adding binary-encoded nonnegative slack variables s and a square penalty:
        minimize x^T Q x + q^T x + P * sum_i (a_i^T x + s_i - b_i)^2
    where s_i is encoded with m_i binary slack bits so that 0..(2^{m_i}-1) covers [0, b_i].

    Returns:
        Q_qubo, q_qubo, offset, for variable order [x (n vars), then all slack bits].
    """
    # convert inputs to arrays
    Q = np.asarray(Q, dtype=float)
    q = np.asarray(q, dtype=float)
    A_le = np.asarray(A_le, dtype=float)
    b_le = np.asarray(b_le, dtype=float)

    # choose a safe default penalty if not supplied
    if penalty is None:
        obj_bound = np.sum(np.abs(Q)) + np.sum(np.abs(q))
        penalty = 1000.0 * obj_bound

    n = Q.shape[0]
    m_cons = A_le.shape[0]

    # decide slack encoding sizes per-constraint (binary encoding 0..b_i)
    slack_bits_per_con = []
    total_slack_bits = 0
    for i in range(m_cons):
        ub = int(round(b_le[i]))
        m_i = _binary_slack_bits(ub)
        slack_bits_per_con.append(m_i)
        total_slack_bits += m_i

    N = n + total_slack_bits
    Q_out = np.zeros((N, N), dtype=float)
    q_out = np.zeros(N, dtype=float)
    offset = 0.0

    # carry original objective
    Q_out[:n, :n] += Q
    q_out[:n] += q

    # build penalty contributions
    slack_cursor = n
    for i in range(m_cons):
        a = A_le[i]             # shape (n,)
        b = float(b_le[i])
        m_i = slack_bits_per_con[i]

        # (a^T x)^2 term
        Q_out[:n, :n] += penalty * np.outer(a, a)

        # -2 b (a^T x) term
        q_out[:n] += penalty * (-2.0 * b) * a

        # + b^2 constant
        offset += penalty * (b ** 2)

        if m_i > 0:
            # binary-encoded slack s = sum_k 2^k y_k
            w = np.array([2 ** k for k in range(m_i)], dtype=float)

            # s^2 term: diag and off-diag on slack bits
            for k in range(m_i):
                idx_k = slack_cursor + k
                Q_out[idx_k, idx_k] += penalty * (w[k] ** 2)
                for l in range(k + 1, m_i):
                    idx_l = slack_cursor + l
                    Q_out[idx_k, idx_l] += penalty * (2.0 * w[k] * w[l])

            # cross term 2(a^T x) s -> x_j y_k
            for j in range(n):
                aj = a[j]
                if aj == 0.0:
                    continue
                for k in range(m_i):
                    idx_k = slack_cursor + k
                    lo, hi = (j, idx_k) if j < idx_k else (idx_k, j)
                    Q_out[lo, hi] += 2.0 * penalty * aj * w[k]

            # linear term -2 b s -> on slack bits
            for k in range(m_i):
                idx_k = slack_cursor + k
                q_out[idx_k] += penalty * (-2.0 * b) * w[k]

        slack_cursor += m_i

    return Q_out, q_out, float(offset)



def qubo_to_ising(
    Q: np.ndarray,
    q: np.ndarray,
    offset_qubo: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Convert QUBO over x in {0,1} to Ising over z in {-1,+1} via x = (z+1)/2:
      min x^T Q x + q^T x + c  ->  min z^T J z + h^T z + const

    Returns:
      h (N,), J (NxN symmetric; upper triangle used), offset (float)
    """
    Q = np.asarray(Q, dtype=float)
    q = np.asarray(q, dtype=float)
    N = Q.shape[0]

    # Expand:
    # x^T Q x = 1/4 z^T Q z + 1/2 z^T Q 1 + 1/4 1^T Q 1
    # q^T x   = 1/2 q^T z + 1/2 q^T 1
    ones = np.ones(N)

    J = Q / 4.0
    h = 0.5 * (Q @ ones) / 2.0 + 0.5 * q
    const = offset_qubo + (ones @ Q @ ones) / 16.0 + 0.5 * (q @ ones)

    J = (J + J.T) / 2.0
    return h.astype(float), J.astype(float), float(const)


if __name__ == "__main__":
    # tiny smoke test using the demo problem
    from .problem_demo import build_demo_problem

    Q, q, A, b = build_demo_problem(8)
    Qq, qq, c = penalize_le_constraints_to_qubo(Q, q, A, b, penalty=1000.0)
    h, J, off = qubo_to_ising(Qq, qq, c)
    print("QUBO size (incl. slacks):", Qq.shape)
    print("Ising h,J shapes:", h.shape, J.shape, "offset:", off)
