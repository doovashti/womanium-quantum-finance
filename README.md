## Quantum Constrained Optimization with VQE

## Team Name
The Entangled Analyst

## Team Members
Vashti Chowla — WISER Enrollment ID: gst-haDItypHlgZFbhy

## Overview of mathematics behind the problem

## Project Summary
1. Given Problem statement:
There are plenty of real world problems that can be framed as binary quadratic optimization problems with constraints - an example of this being portfolio optimization in finance. However, these problems are known to quickly become computationally expensive for classical solvers as the number of input variables we have increases. This is where quantum computing has been a godsend, because if we use it correctly, it has great potential to explore these large variable spaces efficiently. In this project, we are working with binary decision variables, linear inequality constraints and a quadratic objective function. Mathematically, this takes the form:

\[
\min_{x \in \{0,1\}^n} x^T Q x + q^T x \quad \text{s.t.} \quad A x \le b
\]

where \(Q\) is a quadratic cost matrix, \(q\) is a linear cost vector, and \(A x \le b\) represents linear inequality constraints.  
Such problems are **NP-hard** and quickly become intractable for classical exact solvers as \(n\) grows.  

2. Project goals
In this project, I aimed to rewrite the constrained problem into Quantum Unconstrained Binary Optimization (QUBO) form. Then I wanted to implement a Variational Quantum Eigensolver (VQE) as suggested in the project documentation to solve the problem on a quantum simulator, with the goal of validating results against classical baselines using an exact solver and simulated annealing, and then to compare how the quantum assisted solution fared against its classical counterparts. 

3. Approach
The approach I took to achieve the above goals is as follows: 
QUBO Conversion:
    Introduced binary slack variables to exactly encode linear constraints.
    Added penalty terms to discourage violations.
QUBO to Ising Mapping:
    Transformed QUBO into an Ising Hamiltonian, enabling quantum variational algorithms.
VQE Implementation:
    Used TwoLocal ansatz with RY rotations, full entanglement, and multiple repetitions.
    Optimized parameters using COBYLA from SciPy.
Classical Baselines:
    Exact solver (brute force) for small problem size.
    Simulated annealing for scalable near-optimal solutions.

4. Results
Exact Solver: f = –1.2039, no constraint violations.
Simulated Annealing: Same as exact (feasible).
VQE: f ≈ –1.6976, with small constraint violations.
Quantum pipeline works end-to-end but shows some trade-off between feasibility and optimality due to penalty tuning.
Outputs stored in results/ (JSON + plots).

5. Impact
So, what was the impact of this project? I've built a workflow for quantum optimization with constraints, and demonstrated the feasibility of applying VQE to constrained problems in finance. This repo contains a flexible framework for experimenting with different quantum ansatzes, and compared quantum methods with classical ones in terms of solution quality and feasibility. 

6. Future Scope
I could improve the constraint satisfaction by adaptive penalty scaling. 
Testing QAOA or hybrid quantum-classical approaches.
Benchmark scaling to higher variable sizes. 
Testing on real quantum harware with noise models. 

## Link to presentation


```bash
pip install -r requirements.txt
python main.py --mode quantum
=======
# womanium-quantum-finance
Quantum optimization for bond portfolio selection using QUBO, QAOA, and classical solvers.
