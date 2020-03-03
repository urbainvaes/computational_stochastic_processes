# +
# Copyright (c) 2020 Urbain Vaes. All rights reserved.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import numpy as np
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

G = nx.DiGraph()
G.add_edges_from([('A', 'C')], weight=1)
G.add_edges_from([('B', 'A')], weight=.5)
G.add_edges_from([('B', 'C')], weight=.6)
G.add_edges_from([('C', 'B')], weight=.7)
G.add_edges_from([('C', 'D')], weight=.8)
G.add_edges_from([('D', 'C')], weight=.9)
G.add_edges_from([('D', 'E')], weight=.5)
G.add_edges_from([('E', 'C')], weight=1)

pos = {'A': (0, 0), 'B': (0, 2), 'C': (1, 1), 'D': (2, 0), 'E': (2, 2)}

fig, ax = plt.subplots()

kwargs = {
        'fontsize': 18,
        'horizontalalignment': 'center',
        'verticalalignment': 'center',
        'transform': ax.transAxes,
        }
text = ax.text(.3, .79, "1/2".format(0), **kwargs)
text = ax.text(.3, .62, "1/2".format(0), **kwargs)
text = ax.text(.3, .28, "1".format(0), **kwargs)
text = ax.text(.7, .79, "1".format(0), **kwargs)
text = ax.text(.7, .38, "1/2".format(0), **kwargs)
text = ax.text(.7, .20, "1/2".format(0), **kwargs)

labels={}
labels['A']=r'$a$'
labels['B']=r'$b$'
labels['C']=r'$c$'
labels['D']=r'$d$'
labels['E']=r'$\alpha$'

labels = {'A': 1000, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
values = [labels[node] for node in G.nodes()]

nx.draw_networkx_labels(G, pos, labels, font_size=16)

nx.draw(G, pos, node_color=values, node_size=1500, connectionstyle='arc3, rad=0.1', ax=ax)
# nx.draw(G, pos, node_size=3000, connectionstyle='arc3, rad=0.1', ax=ax)
plt.show()
