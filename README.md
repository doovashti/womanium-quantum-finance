## Quantum Constrained Optimization with VQE

## Team Name
The Entangled Analyst

## Team Members
Vashti Chowla — WISER Enrollment ID: gst-haDItypHlgZFbhy

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

## Link to presentation https://www.canva.com/design/DAGvcNzejoQ/60Zg34Xru9-0-EDixA8MgA/edit?utm_content=DAGvcNzejoQ&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton


## References
Qiskit – Quantum Information Science Kit.
IBM Research. Available at: https://qiskit.org/

NumPy – Fundamental package for scientific computing with Python.
Harris, C.R. et al. (2020). Nature, 585, 357–362.

SciPy – Scientific computing tools for Python.
Virtanen, P. et al. (2020). Nature Methods, 17, 261–272.

Matplotlib – Python 2D plotting library.
Hunter, J.D. (2007). Computing in Science & Engineering, 9(3), 90–95.

Variational Quantum Eigensolver (VQE) – Algorithm overview.
Peruzzo, A. et al. (2014). A variational eigenvalue solver on a quantum processor. Nature Communications, 5, 4213.

QUBO formulation for constrained optimization –
Glover, F., Kochenberger, G., & Du, Y. (2019). Quantum bridge analytics I: a tutorial on formulating and using QUBO models. 4OR, 17, 335–371.


```bash
pip install -r requirements.txt
python main.py --mode quantum
