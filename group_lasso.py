#
# Author: Fabian Pedregosa <fabian@fseoane.net>
# Author: Alejandro Catalina <alecatfel@gmail.com>
# License: BSD
import numpy as np
import group_lasso_fast as gl


class _BaseGroupLasso(object):

    def fit(self, X, y):
        self.coefs = gl.solve_group_lasso(
            X, y, self.alpha, self.l1, self.groups, self.tol, self.verbose)

        return self

    def predict(self, X):
        return np.dot(X, self.coef)


class GroupLasso(_BaseGroupLasso):

    def __init__(self, alpha, groups, max_iter=1000, tol=1e-6, verbose=False):
        self.alpha = alpha
        self.groups = groups
        self.tol = tol
        self.l1 = 0.
        self.verbose = verbose


class SparseGroupLasso(_BaseGroupLasso):

    def __init__(self, alpha, l1, groups, max_iter=1000,
                 tol=1e-6, verbose=False):
        self.alpha = alpha
        self.groups = groups
        self.tol = tol
        self.l1 = l1
        self.verbose = verbose
