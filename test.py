import group_lasso as gl
import numpy as np
from sklearn import datasets

np.random.seed(0)
alpha = 0.1

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
groups = np.arange(X.shape[1]) // 5
clf = gl.GroupLasso(alpha, groups, verbose=True)
clf.fit(X, y)
