# src/classical_solve.py
import numpy as np
from typing import Tuple

def obj_value(Q: np.ndarray, q: np.ndarray, x: np.ndarray) -> float:
    return float(x @ Q @ x + q @ x)

def violations(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.maximum(A @ x - b, 0.0)

def brute_force(Q, q, A, b) -> Tuple[np.ndarray, float, np.ndarray]:
    n = len(q)
    best_x, best_val, best_v = None, float("inf"), None
    for mask in range(1 << n):
        x = np.array([(mask >> i) & 1 for i in range(n)], dtype=float)
        v = violations(A, b, x)
        if np.all(v == 0):
            val = obj_value(Q, q, x)
            if val < best_val:
                best_x, best_val, best_v = x, val, v
    return best_x, best_val, best_v

def simulated_annealing(Q, q, A, b, steps=20000, T0=2.0, alpha=0.999) -> Tuple[np.ndarray, float, np.ndarray]:
    rng = np.random.default_rng(42)
    n = len(q)
    x = rng.integers(0, 2, size=n).astype(float)
    def penalized(x):
        return obj_value(Q, q, x) + 1000.0 * np.sum(violations(A, b, x)**2)
    fx = penalized(x)
    best_x, best_fx = x.copy(), fx
    T = T0
    for _ in range(steps):
        i = rng.integers(0, n)
        x[i] = 1.0 - x[i]
        f_new = penalized(x)
        d = f_new - fx
        if d < 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            fx = f_new
            if f_new < best_fx:
                best_fx, best_x = f_new, x.copy()
        else:
            x[i] = 1.0 - x[i]
        T *= alpha
    v = violations(A, b, best_x)
    return best_x, obj_value(Q, q, best_x), v
