# Motivation

CVXPY is a very flexible modelling language for solving convex optimization problems in Python. Its API offers users the 
ability to model mathematical optimization problems very intuitively, and supports numerous solvers that can handle LP, QP, 
SOCP, SDP, EXP and MIP programs. The challenge for the user is to ensure that their program, if possible, is formulated such 
that it follows DCP (Disciplined Convex Programming) rules. Many problems in practice can be formulated in convex form at the 
outset, while others do not lend themselves to such an obvious formulation. Here we consider a problem of MINLP form, 
demonstrate its reformulation into MILP form through the use of variable relaxations and constraint additions. We then model 
the program in CVXPY and call the open source GLPK_MI solver. In particular, the non-linearity comes from the product of 
continuous and binary variables J=X*Z which must be reformulated in the form J'=X'+Z' to satisfy DCP rules. In addition, 
we deal with Boolean variables Y which are implicitly determined by the values of the continuous decision variables Y≔f(X).

# Installation

```{r}
git clone https://github.com/pstarszyk/solving_MINLP_in_CVXPY.git
cd solving_MINLP_in_CVXPY
pip install -r requirements.txt
```

# Usage
Please see the detailed example in `problem.pdf` with implementation in `implementation.py`. In order to use `GLPK_MI` solver, 
`cvxopt` must be installed.

For more detailed information on the CVXPY package, please refer to the documentation.

```{r}
Steven Diamond and Stephen Boyd, CVXPY: A Python-Embedded Modeling Language for Convex Optimization
https://buildmedia.readthedocs.org/media/pdf/cvxpy/latest/cvxpy.pdf
```

