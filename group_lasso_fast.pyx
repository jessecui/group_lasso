# encoding: utf-8
# cython: cdivision=True
#
# Author: Fabian Pedregosa <fabian@fseoane.net>
# License: BSD
import numpy as np
cimport numpy as np
from scipy import linalg


DEF MAX_ITER = 1000


cdef soft_threshold(np.ndarray[double, ndim=1] a,
                    double b):
    # vectorized version
    return np.sign(a) * np.fmax(np.abs(a) - b, 0)


cpdef sparse_group_lasso(np.ndarray[double, ndim=2] X,
                         np.ndarray[double] y,
                         double alpha,
                         double rho,
                         np.ndarray[long] groups,
                         int max_iter=MAX_ITER,
                         double rtol=1e-6,
                         int verbose=0):
    """
    Linear least-squares with l2/l1 + l1 regularization solver.

    Solves problem of the form:

    (1 / (2 n_samples)) * ||Xb - y||^2_2 +
        [ (alpha * (1 - rho) * sum(sqrt(#j) * ||b_j||_2) + alpha * rho ||b_j||_1) ]

    where b_j is the coefficients of b in the
    j-th group. Also known as the `sparse group lasso`.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Design Matrix.

    y : array of shape (n_samples,)

    alpha : float or array
        Amount of penalization to use.

    rho : float
        Amount of l1 penalization

    groups : array of shape (n_features,)
        Group label. For each column, it indicates
        its group apertenance.

    rtol : float
        Relative tolerance. ensures ||(x - x_) / x_|| < rtol,
        where x_ is the approximate solution and x is the
        true solution. TODO duality gap

    Returns
    -------
    coef : array
        vector of coefficients

    References
    ----------
    "A sparse-group lasso", Noah Simon et al.
    """
    # .. local variables ..
    cdef np.ndarray[double, ndim = 1] w_new = np.zeros(X.shape[1], dtype=X.dtype)
    cdef int n_samples = X.shape[0]
    cdef int i, p, n_iter

    # .. use integer indices for groups ..
    cdef np.ndarray[long, ndim = 2] group_labels = np.array([np.where(groups == i)[0] for i in np.unique(groups)])
    cdef np.ndarray[double, ndim = 1] Xy = np.dot(X.T, y)
    cdef np.ndarray[double, ndim = 2] K = np.dot(X.T, X)
    cdef double step_size = 1. / (linalg.norm(X, 2) ** 2)
    cdef np.ndarray[double, ndim = 3] _K = np.array([K[group][:, group] for group in group_labels])
    alpha = np.asanyarray(alpha)
    alpha = alpha * n_samples
    cdef np.ndarray[double, ndim = 1] w_old, X_residual, X_r_k, s, tmp
    cdef np.ndarray[long, ndim = 1] perm
    cdef double delta, p_j, norm_w_new
    cdef np.ndarray[double, ndim = 2] Kgg

    for n_iter in xrange(max_iter):
        w_old = w_new.copy()
        perm = np.random.permutation(len(group_labels))
        # could be updated, but kernprof says it's peanuts
        # .. step 1 ..
        X_residual = Xy - np.dot(K, w_new)
        for p in xrange(len(perm)):
            i = perm[p]
            group = group_labels[i]
            p_j = np.sqrt(group.size)
            Kgg = _K[i]
            X_r_k = X_residual[group] + np.dot(Kgg, w_new[group])
            s = soft_threshold(X_r_k, alpha * rho)
            # .. step 2 ..
            if np.linalg.norm(s) <= (1 - rho) * alpha * p_j:
                w_new[group] = 0.
            else:
                # .. step 3 ..
                for _ in xrange(2 * group.size):  # just a heuristic
                    grad_l = - (X_r_k - np.dot(Kgg, w_new[group]))
                    tmp = soft_threshold(
                        w_new[group] - step_size * grad_l, step_size * rho * alpha)
                    tmp *= max(1 - step_size * p_j * (1 - rho)
                               * alpha / np.linalg.norm(tmp), 0)
                    delta = linalg.norm(tmp - w_new[group])
                    w_new[group] = tmp
                    if delta < 1e-3:
                        break

                assert np.isfinite(w_new[group]).all()

        norm_w_new = max(np.linalg.norm(w_new), 1e-10)
        if np.linalg.norm(w_new - w_old) / norm_w_new < rtol:
            break
    return w_new
