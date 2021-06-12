""" Auxiliary model-related functions. """

import numpy as np
from scipy.special import erf
from scipy.linalg import eig, inv


def perform_gsr(cov):
    """
    Perform global signal regression on a covariance matrix.

    Parameters
    ----------
    cov : np.ndarray
        covariance matrix

    Returns
    -------
    np.ndarray
        global signal-regressed covariance matrix
    """
    gbc_cov = cov.sum(axis=0)
    # print(type(cov), type(gbc_cov))
    gsr_cov = cov - np.multiply.outer(gbc_cov, gbc_cov) / cov.sum()
    return gsr_cov


def solve_lyapunov_fast(A, Q, B=None):
    """
    Solve Lyapunov equation fast using eigendecomposition.

    Parameters
    ----------
    A : np.ndarray
        Jacobian matrix with dimensions (N, N)
    Q : np.ndarray or scipy.sparse.csc_matrix
        noise covariance matrix with dimensions (N, N)
    B : np.ndarray (optional) with dimensions (1, N)
        (intended for computing BOLD covariance) a matrix containing the
         partial derivatives of some other quantity, Y, with respect to the
         variables in each column of A and Q. That is, if A is the Jacobian
         matrix for some quantity X, then B_ij = dY_i/dX_j.

    Returns
    -------
    P : np.ndarray
        covariance matrix

    Notes
    -----
    Solves the Lyapunov equation AP + PA' = Q. This method is ~100x faster
    than the built-in scipy.linalg.solve_lyapunov solver. The `B` argument
    should be used when solving for the covariance of `Y` in the following
    system of equations:
        dX/dt = A X
        Y = B X
    IMPORTANT: Significant speed increases are possible if A, B, and Q have
    dtype == np.float32, rather than np.float64.

    """
    # evals, L = eig(A.T, left=True, right=False)
    evals, L = eig(A, left=False, right=True)
    L_inv = inv(L)
    Qt = L_inv.dot(-Q.dot(L_inv.conj().T)) / np.add.outer(evals, evals.conj())
    if B is not None:
        bL = B.dot(L)
        return bL.dot(Qt.dot(bL.conj().T)).real
    return L.dot(Qt.dot(L.conj().T)).real


def load_model_params():
    """ 
    Returns the model's synaptic parameters defined
    in synaptic.py as a python dictionary.
    """
    from .params import synaptic
    params = clean_builtins(vars(synaptic))
    return params


def clean_builtins(my_dict):
    """
    Cleans the dictionaries
    """

    cleaned_dict = dict()
    for key, val in my_dict.items():
        if key[:2] != "__":
            cleaned_dict[key] = val
    return cleaned_dict


def cov_to_corr(cov):
    """ 
    Generate correlation matrix from covariance matrix.
    """
    cov_ii = cov_jj = np.diag(cov).astype(np.float64)
    return cov / np.sqrt(np.outer(cov_ii, cov_jj))


def linearize_map(x):
    """
    linearization function for T1w/T2w maps
    """
    return erf((x - np.mean(x)) / x.std() / np.sqrt(2))
