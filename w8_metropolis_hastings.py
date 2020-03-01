# +
# Copyright (c) 2020 Urbain Vaes. All rights reserved.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import numpy as np
import scipy.stats
import scipy.integrate
import scipy.optimize
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# -

# +
matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('figure', figsize=(16, 11))
matplotlib.rc('lines', linewidth=2)
matplotlib.rc('lines', markersize=12)
matplotlib.rc('figure.subplot', hspace=.3)
matplotlib.rc('figure.subplot', wspace=.1)
matplotlib.rc('animation', html='html5')
np.random.seed(0)

