#
# Author: Fabian Pedregosa <fabian@fseoane.net>
# Author: Alejandro Catalina <alecatfel@gmail.com>
# License: BSD
import numpy as np
import group_lasso_fast as gl


class _BaseGroupLasso(object):

    def fit(self, X, y):
        self.coef_ = gl.solve_group_lasso(X, y, self.alpha, self.l1,
                                          self.groups, self.max_iter, self.tol)

        return self

    def predict(self, X):
        return np.dot(X, self.coef)


class GroupLasso(_BaseGroupLasso):

    def __init__(self, alpha=1., groups=[], max_iter=1000, tol=1e-6):
        self.alpha = alpha
        self.groups = groups
        self.tol = tol
        self.l1 = 0.
        self.max_iter = max_iter


class SparseGroupLasso(_BaseGroupLasso):

    def __init__(self, alpha=1.0, l1=0., groups=[], max_iter=1000, tol=1e-6):
        self.alpha = alpha
        self.groups = groups
        self.tol = tol
        self.max_iter = max_iter
        self.l1 = l1
