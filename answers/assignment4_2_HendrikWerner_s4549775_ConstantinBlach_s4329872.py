# author: Hendrik Werner s4549775
# author: Constantin Blach s4329872

# assignment 4.2.1
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
from scipy.io import loadmat

from packages.clusterPlot import clusterPlot

synth1 = loadmat("./data/synth1.mat")
synth2 = loadmat("./data/synth2.mat")
synth3 = loadmat("./data/synth3.mat")
synth4 = loadmat("./data/synth4.mat")
synths = [synth1, synth2, synth3, synth4]


def plot_cluster_dendrogram(data: np.ndarray, method: str, data_name: str):
    X = data["X"]
    y = data["y"]
    Z = linkage(X, method=method)
    dendrogram(Z, truncate_mode="lastp", p=4)
    plt.title("{} Dendrogram (method: {})".format(data_name, method))
    plt.xlabel("data points")
    plt.ylabel("{} closeness".format(method))
    plt.show()

    clusters = fcluster(Z, 4, criterion="maxclust")
    clusters = clusters - np.ones(clusters.shape)
    plt.scatter(X[:, 0], X[:, 1], c=clusters)
    clusterPlot(X, clusters, y=y)
    plt.title("{} Clusters (method: {})".format(data_name, method))
    plt.xlabel("attribute 1")
    plt.ylabel("attribute 2")
    plt.show()

for linkage_method in ["single", "complete", "average"]:
    for i, synth in enumerate(synths):
        plot_cluster_dendrogram(synth, linkage_method, "Synth{}".format(i + 1))
