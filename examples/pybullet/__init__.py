# pybullet-planning (the examples/pybullet/utils submodule) still calls np.math.atan2, but NumPy 2.0 removed the
# np.math alias of the standard math module. Restore it here so the pinned submodule works without a fork.
import math

import numpy

if not hasattr(numpy, 'math'):
    numpy.math = math
