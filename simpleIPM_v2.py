""" Copyright 2017, Sascha-Dominic Schnug, All rights reserved.
    Following: Numerical Optimization by Nocedal, Wright """


from __future__ import division
import itertools
import six
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splin


def lp_standardization_v4(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None):
    """
    """
    rows_EQ, cols_EQ = None, None
    rows_UB, cols_UB = None, None
    if A_eq is not None:
        rows_EQ, cols_EQ = A_eq.shape
    if A_ub is not None:
        rows_UB, cols_UB = A_ub.shape

    n_orig_vars = len(c)

    assert (A_ub is not None) or (A_eq is not None)

    if (A_ub is None) and (A_eq is not None):
        # only A_eq -> needs no change (assuming x >= 0 for all vars)
        return c, A_eq.tocsc(), b_eq

    if (A_ub is not None) and (A_eq is None):
        # only A_ub
        c = np.hstack((c, -c, np.zeros(rows_UB)))
        A = sp.hstack((A_ub, -A_ub, sp.identity(rows_UB)))
        b = b_ub
        return c, A.tocsc(), b

    if (A_ub is not None) and (A_eq is not None):
        # both!
        c = np.hstack((c, -c, np.zeros(rows_UB)))
        A_upper = sp.hstack((A_ub, -A_ub, sp.identity(rows_UB)))
        A_lower = sp.hstack((A_eq, -A_eq, sp.csc_matrix((rows_EQ, rows_UB))))
        A = sp.vstack((A_upper, A_lower))
        b = np.hstack((b_ub, b_eq))
        return c, A.tocsc(), b

class LP_IPM_Solver(object):
    """
        Expects:
            min  c^T x
            s.t. Ax = b,
                 x >= 0 (if not free)
                 free vars marked
    """
    def __init__(self, c, A_ub=None, b_ub=None,
                          A_eq=None, b_eq=None,
                          maxiter=150, disp=False, prefer_umfpack=True,
                          dyn_reg_eps=1e-9):
        self.max_iter = maxiter
        self.dyn_reg_eps = dyn_reg_eps

        self.c = c
        if not sp.issparse(A_ub):
            self.A_ub = sp.csc_matrix(A_ub)
        else:
            self.A_ub = A_ub
        self.b_ub = b_ub
        if not sp.issparse(A_eq):
            self.A_eq = sp.csc_matrix(A_eq)
        else:
            self.A_eq = A_eq
        self.b_eq = b_eq
        self.maxiter = maxiter
        self.disp = disp

        self.standardize()

        # check if umfpack is available
        self.USE_UMFPACK = False
        if prefer_umfpack:
            try:
                from scikits.umfpack import splu as umfsplu
                #umfsplu
                self.USE_UMFPACK = True
                print('use umfpack')
            except ImportError:
                print('use superlu')
                pass

    def standardize(self):
        """ Not 100% standard form as there might be free-vars """
        self.standard_c, self.standard_A, self.standard_b = \
            lp_standardization_v4(self.c, self.A_ub, self.b_ub, self.A_eq, self.b_eq)

    @staticmethod
    def find_initial_solution(A, b, c):
        """ Solve min 0.5dot(x,x) s.t. A x = b
            -> Least-norm solution via QR factorization
               (https://see.stanford.edu/materials/lsoeldsee263/08-min-norm.pdf)
        """
        AAT = A.dot(A.T)
        AAT_fact = splin.splu(AAT)
        mu = AAT_fact.solve(b)
        x_tilde = A.T.dot(mu)

        """ Solve min 0.5dot(s,s) s.t. A'lamda + s = c
            -> default IanNZ
            WORKS!
        """
        lambda_tilde = AAT_fact.solve(A.dot(c))
        s_tilde = c - A.T.dot(lambda_tilde)

        """ Some components of x,s maybe be negative
            Add constant offset to all elements to ensure non-negativity
            WORKS!
        """
        delta_x = max(-3/2 * np.amin(x_tilde), 0)
        delta_s = max(-3/2 * np.amin(s_tilde), 0)
        x_hat = x_tilde + delta_x
        s_hat = s_tilde + delta_s

        """ Now ensure that the components are not too close to zero and not too
            dissimilar, add two more scalars and return our starting point
            WORKS!
        """
        delta_hat_x = 1/2 * x_hat*s_hat / np.sum(s_hat)
        delta_hat_s = 1/2 * x_hat*s_hat / np.sum(x_hat)

        return x_hat + delta_hat_x, lambda_tilde, s_hat + delta_hat_s

    def solve(self):
        """ might be broken for superLU """

        if self.USE_UMFPACK:
            import scikits.umfpack as um

        umfpack = um.UmfpackContext() # Use default 'di' family of UMFPACK routines.
        umfpack.control[um.UMFPACK_STRATEGY_SYMMETRIC] = True
        umfpack.control[um.UMFPACK_PRL] = 0  # not working

        # UGLY
        c = self.standard_c
        A = self.standard_A
        b = self.standard_b

        M, N = A.shape
        converged = False

        """ Initial solution """
        x, lambda_, s = self.find_initial_solution(A, b, c)
        zero_m_m = sp.csc_matrix((M,M))

        iter = 1
        while (not converged):
            if iter > self.max_iter:
                return False, np.nan

            """ Affine-scaling directions: 14.41 general form """
            D = sp.diags(np.sqrt(1/s)).dot(sp.diags(np.sqrt(x)))  # CORRECT
            D_ = -sp.diags(1.0 / np.square(D.diagonal()))  # CORRECT (octave test)

            diag = D_.diagonal()
            pos_inds = np.where(diag >= 0.0)
            neg_inds = np.where(diag < 0.0)

            new_diag = diag[:]
            new_diag[pos_inds] += self.dyn_reg_eps
            new_diag[neg_inds] -= self.dyn_reg_eps
            D_.setdiag(new_diag)

            lhs = sp.bmat([[D_, A.T],
                           [A, zero_m_m]], format='csc')

            lhs_fact = None
            if self.USE_UMFPACK:
                if iter == 1:
                    umfpack.symbolic(lhs)
                    #umfpack.report_symbolic()
                umfpack.numeric(lhs)
                #umfpack.report_numeric()

            else:
                lhs_fact = splin.splu(lhs)#, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0, options=dict(SymmetricMode=True)) # superlu

            r_b = A.dot(x) - b
            r_c = A.T.dot(lambda_) + s - c
            r_xs = x*s
            rhs = np.hstack([-r_c + sp.diags(1/x).dot(r_xs), -r_b])

            sol = None
            if self.USE_UMFPACK:
                sol = umfpack(um.UMFPACK_A, lhs, rhs, autoTranspose = True )
            else:
                sol = lhs_fact.solve(rhs)

            delta_x_aff = sol[:N]
            delta_lambda_aff = sol[N:N+M]

            delta_s_aff = -sp.diags(1/x).dot(r_xs) - (sp.diags(1/x).dot(sp.diags(s))).dot(delta_x_aff)

            """ Affine-scaling step-length: 14:32 + 14:33 """
            alpha_pri_aff = 1.0
            for i in range(N):
                if delta_x_aff[i] < 0.0 and not np.isclose(delta_x_aff[i], 0.0):  # CRITICAL
                    alpha_pri_aff = min(alpha_pri_aff, -x[i] / delta_x_aff[i])

            alpha_dual_aff = 1.0
            for i in range(M):
                if delta_s_aff[i] < 0.0 and not np.isclose(delta_x_aff[i], 0.0):  # CRITICAL
                    alpha_dual_aff = min(alpha_dual_aff, -s[i] / delta_s_aff[i])

            mu = 1/N * np.dot(x, s)
            mu_aff = 1/N * np.dot(x + alpha_pri_aff * delta_x_aff, s + alpha_dual_aff * delta_s_aff)

            """ Centering param """
            sigma = (mu_aff / mu)**3

            """ Re-solve for directions """
            r_xs_ = r_xs + delta_x_aff*delta_s_aff - sigma*mu

            # CRITICAL # TODO
            rhs = np.hstack([-r_c + sp.diags(1/x).dot(r_xs_), -r_b])

            sol_ = None
            if self.USE_UMFPACK:
                sol_ = umfpack(um.UMFPACK_A, lhs, rhs, autoTranspose = True )
            else:
                sol_ = lhs_fact.solve(rhs)

            delta_x = sol_[:N]
            delta_lambda = sol_[N:N+M]
            delta_s = -sp.diags(1/x).dot(r_xs_) - (sp.diags(1/x).dot(sp.diags(s))).dot(delta_x)

            """ Step-lengths """
            eta = 0.9
            alpha_pri_max = np.inf
            for i in range(N):
                if delta_x[i] < 0.0:
                    alpha_pri_max = min(alpha_pri_max, -x[i] / delta_x[i])
            alpha_dual_max = np.inf
            for i in range(N):
                if delta_s[i] < 0.0:
                    alpha_dual_max = min(alpha_dual_max, -s[i] / delta_s[i])
            alpha_pri  = min(1.0, eta*alpha_pri_max)
            alpha_dual = min(1.0, eta*alpha_dual_max)

            """ Update current solution """
            x += alpha_pri * delta_x
            lambda_ += alpha_dual * delta_lambda
            s += alpha_dual * delta_s

            if abs(np.dot(c, x) - np.dot(b, lambda_)) < 1e-5:
                converged=True
            else:
                iter += 1

        return True, np.dot(c,x)
