This is prototype-code to analyze/evaluate some LP-solvers already in or designed for
scipy.

This code, in some other form, was previously used privately to debug and benchmark
an own Interior-Point-Solver for Linear-Programming. Some version of this code
(probably in bad shape) is withint this repo.

# Overview
This code uses [cvxopt's](http://cvxopt.org/index.html) MPS-parser to prepare and
run netlib-instances with different solvers.

# Setup / Requirements
## Subfolder ```netlib_instances```
- Put uncompressed .mps instances here
- Add them with the given objective (see readme) to ```constants.py```
- (Not within this repos because of possible license-issues)

## Packages / Modules
Core:
  - numpy, scipy, cython

MPS-Reading and Canonicalization:
  - cvxopt

Data-collecting (SQL-based now):
  - dataset

Evaluation and Plotting:
  - matplotlib
  - seaborn
  - pandas

3rd-party:
  - scikit-umfpack (only needed for simpleIPM; superLU broken in current version)

# Instances supported
- Only netlib-instances were tested
- Only instances without range-constraints where tested
    - see [netlib-readme](http://www.netlib.org/lp/data/readme) for table of instance-types

# Solvers supported
- scipy.optimize.linprog method='simplex': ```linprog_simplex```
- scipy.optimize.linprog method='interior-point' (unmerged): ```linprog_ip```
- cvxopt's conelp: ```cvxopt_conelp```
- cvxopt + mosek: **broken**
- own simpleIPM: ```simpleIPM```

# Todo / Problems / Ugly-stuff
- Benchmark-principles are not well-designed
 - Currently:
   - raise Error: failed instance!
   - warning (like "no convergence detected"): success if ```result == expected-result (within tol)```
- Repair Mosek
- Remove unwanted verbosity
  - cvxopt's preprocessing
  - Mosek-based solving
  - scikit-umfpack: condition-number warnings
