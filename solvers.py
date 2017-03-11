import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from cvxopt.solvers import lp
from cvxopt.base import matrix, spmatrix
from cvxopt.solvers import options
from simpleIPM_v2 import LP_IPM_Solver

options['show_progress'] = False
options['MOSEK'] = {'mosek.iparam.log': 0}  # TODO: not working

""" utils """
def scipy_sparse_to_spmatrix(A):
    A = sp.csc_matrix(A)  # hack
    coo = A.tocoo()
    SP = spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
    return SP

""" solver-functions """
def solve_linprog_simplex(problem):
    c, G, h, A, b = problem
    G = sp.csc_matrix(G)
    A = sp.csc_matrix(A)
    try:
        result = linprog(c, G.todense(), h, A.todense(), b)
    except:
        result = False
        status, objective = False, np.nan

    if result:
        status, objective = result.status, result.fun

    return status, objective

def solve_linprog_ip(problem):
    c, G, h, A, b = problem
    G = sp.csc_matrix(G)
    A = sp.csc_matrix(A)
    try:
        result = linprog(c, G.todense(), h, A.todense(), b, method='interior-point')  # TODO: sparse?
    except:
        result = False
        status, objective = False, np.nan

    if result:
        status, objective = result.status, result.fun

    return status, objective

# TODO refactor!
def solve_cvxopt_conelp(problem, solver='conelp'):
    c, G, h, A, b = problem
    c = matrix(c)
    G = scipy_sparse_to_spmatrix(G)
    h = matrix(h)
    A = scipy_sparse_to_spmatrix(A)
    b = matrix(b)
    try:
        result = lp(c, G, h, A, b, solver=solver)
    except:
        result = False
        status, objective = False, np.nan

    if result:
        status, objective = result['status'], result['primal objective']

    return status, objective

def solve_cvxopt_mosek(problem, solver='mosek'):
    # TODO: something wrong while returning
    c, G, h, A, b = problem
    c = matrix(c)
    G = scipy_sparse_to_spmatrix(G)
    h = matrix(h)
    A = scipy_sparse_to_spmatrix(A)
    b = matrix(b)
    try:
        result = lp(c, G, h, A, b, solver=solver)
    except:
        result = False
        status, objective = False, np.nan

    if result:
        status, objective = result['status'], result['primal objective']

    return status, objective

def solve_simpleIPM(problem):
    c, G, h, A, b = problem
    G = sp.csc_matrix(G)
    A = sp.csc_matrix(A)
    solver = LP_IPM_Solver(c, G, h, A, b)  # sparse
    try:
        status, objective = solver.solve()
    except:
        result = False
        status, objective = False, np.nan

    return status, objective
