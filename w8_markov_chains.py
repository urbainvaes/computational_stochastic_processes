# +
# Copyright (c) 2020 Urbain Vaes. All rights reserved.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# import time
import numpy as np
import scipy.stats
import networkx as nx
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

# T is the transition matrix
def run_tests(T, action='plot_evolution'):

    G = nx.DiGraph()
    for i, v in enumerate(T):
        for j, n in enumerate(v):
            if n != 0:
                G.add_edges_from([(i, j)], weight=n)

    pos = {0: (0, 0), 1: (0, 2), 2: (1, 1), 3: (2, 0), 4: (2, 2)}

    def add_edges_labels(ax):
        kwargs = {
                'fontsize': 18,
                'horizontalalignment': 'center',
                'verticalalignment': 'center',
                'transform': ax.transAxes,
                }

        if T[1][2] != 0:
            text = ax.text(.3, .62, "{}".format(T[1][2]), **kwargs)
        text = ax.text(.05, .5, "{}".format(T[1][0]), **kwargs)

        if T[3][2] != 0:
            text = ax.text(.7, .38, "{}".format(T[3][2]), **kwargs)
        text = ax.text(.95, .5, "{}".format(T[3][4]), **kwargs)

        text = ax.text(.3, .79, "0.5", **kwargs)
        text = ax.text(.3, .28, "1", **kwargs)
        text = ax.text(.7, .79, "1", **kwargs)
        text = ax.text(.7, .20, "0.5", **kwargs)

    # Number of "particles"
    N = 10**4

    # Number of iterations
    n = 100

    # Number of nodes
    K = len(T)

    # values[i] contains the number of particles at the nodes at iteration i
    values = np.zeros((n + 1, K), dtype=int)
    exact = np.zeros((n + 1, K))
    values[0] = [N, 0, 0, 0, 0]
    exact[0] = [1, 0, 0, 0, 0]
    tr = np.array(T)

    # Generalized Bernoulli distribution for each node
    gen_bernoulli = scipy.stats.rv_discrete
    draw_next = [gen_bernoulli(values=(range(K), v)) for v in T]

    # Simulation of the Markov chain
    for i in range(n):
        for j, v in enumerate(T):
            next_step = draw_next[j].rvs(size=values[i][j])
            for k in next_step:
                values[i+1][k] += 1
        exact[i+1] = tr.T.dot(exact[i])

    def plot_evolution(i):
        ax.clear()
        add_edges_labels(ax)
        labels = {j: v for j, v in enumerate(values[i])}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=16, ax=ax)
        cmap = matplotlib.cm.get_cmap('viridis')
        nx.draw(G, pos, node_color=values[i], alpha=.5, node_size=3000,
                connectionstyle='arc3, rad=0.1', ax=ax, cmap=cmap)
        ax.set_title("Discrete time: ${}$".format(i))

    def plot_pmf(i):
        ax.clear()
        ax.set_title("Probability mass function at iteration ${}$".format(i))
        ax.set_xlabel("Node index")
        ax.stem(np.arange(K) - .05, values[i]/N, use_line_collection=True,
                label="MC approximation", linefmt='C0-', markerfmt='C0o')
        ax.stem(np.arange(K) + .05, exact[i], use_line_collection=True,
                label="Exact", linefmt='C1-', markerfmt='C1o')
        ax.set_ylim(0, 1.1)
        ax.legend()

    # Create animation
    matplotlib.rc('figure', figsize=(12, 8))
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.1, bottom=.1, right=.98, top=.95)
    iterate = plot_evolution if action == 'plot_evolution' else plot_pmf
    anim = animation.FuncAnimation(fig, iterate, np.arange(n),
                                   init_func=lambda: None, repeat=True)
    # For Python
    plt.show()

    # For notebook
    # plt.close(fig)
    # return anim

# -

# +
T = [[0, 0, 1, 0, 0], [1, 0, 0, 0, 0],
     [0, .5, 0, .5, 0], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0]]
run_tests(T, action='plot_evolution')
# -

# +
run_tests(T, action='plot_pmf')
# -

# +
T = [[0, 0, 1, 0, 0], [.5, 0, .5, 0, 0],
     [0, .5, 0, .5, 0], [0, 0, .5, 0, .5], [0, 0, 1, 0, 0]]
run_tests(T, action='plot_evolution')
# -

# +
run_tests(T, action='plot_pmf')
# -

# +
T = [[0, 0, 1, 0, 0], [1, 0, 0, 0, 0],
     [0, .5, 0, .5, 0], [0, 0, .5, 0, .5], [0, 0, 1, 0, 0]]
run_tests(T, action='plot_evolution')
# -

# +
run_tests(T, action='plot_pmf')
# -
