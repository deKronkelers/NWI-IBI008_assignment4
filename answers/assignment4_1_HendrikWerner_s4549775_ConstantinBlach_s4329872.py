# author: Hendrik Werner s4549775
# author: Constantin Blach s4329872

import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.cluster import k_means

from packages.clusterPlot import clusterPlot

# assignment 4.1.1
synth1 = loadmat("./data/synth1.mat")
synth2 = loadmat("./data/synth2.mat")
synth3 = loadmat("./data/synth3.mat")
synth4 = loadmat("./data/synth4.mat")
synths = [synth1, synth2, synth3, synth4]


def plot_clustering(data, clusters: int):
    X = data["X"]
    y = data["y"]
    centroids, labels, inertia = k_means(X, clusters)
    clusterPlot(X, labels, centroids, y)
    plt.show()


for synth in synths:
    plot_clustering(synth, 4)

# assignment 4.1.2

# assignment 4.1.3

# assignment 4.1.4
