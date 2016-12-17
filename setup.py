from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import scipy

setup(
    name='SparseGroupLasso',
    ext_modules=[
        Extension('group_lasso_fast', ['group_lasso_fast.pyx'],
                  include_dirs=[np.get_include(), scipy.get_include()])
    ],
    cmdclass={'build_ext': build_ext}
)
