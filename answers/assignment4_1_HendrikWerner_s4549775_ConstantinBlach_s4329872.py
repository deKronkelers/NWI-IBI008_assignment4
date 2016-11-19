# author: Hendrik Werner s4549775
# author: Constantin Blach s4329872

import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.cluster import k_means

from packages.clusterPlot import clusterPlot
from packages.clusterVal import clusterVal

# assignment 4.1.1
synth1 = loadmat("./data/synth1.mat")
synth2 = loadmat("./data/synth2.mat")
synth3 = loadmat("./data/synth3.mat")
synth4 = loadmat("./data/synth4.mat")
synths = [synth1, synth2, synth3, synth4]


def plot_clustering(
        data, clusters: int,
        title: str = None,
        xlabel: str = None,
        ylabel: str = None
):
    X = data["X"]
    y = data["y"]
    centroids, labels, inertia = k_means(X, clusters)
    clusterPlot(X, labels, centroids, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


for i, synth in enumerate(synths):
    plot_clustering(synth, 4, title="Synth {}".format(i + 1), xlabel="attribute 1", ylabel="attribute 2")

# assignment 4.1.2
def plot_errors(
        data, range,
        title: str = None,
        xlabel: str = None,
        ylabel: str = None
):
    X = data["X"]
    y = data["y"]
    entropies = []
    purities = []
    rands = []
    jaccards = []
    for k in range:
        _, labels, _ = k_means(X, k)
        entropy, purity, rand, jaccard = clusterVal(y, labels)
        entropies.append(entropy)
        purities.append(purity)
        rands.append(rand)
        jaccards.append(jaccard)
    f = plt.subplot(111, title=title, xlabel=xlabel, ylabel=ylabel)
    f.plot(range, entropies, color="r", label="Entropy")
    f.plot(range, purities, color="g", label="Purity")
    f.plot(range, rands, color="b", label="Rand")
    f.plot(range, jaccards, color="black", label="Jaccard")
    plt.legend()
    plt.show()


for i, synth in enumerate(synths):
    plot_errors(synth, range(1, 11), "Synth {}".format(i + 1), "k", "validity measure")

# assignment 4.1.3

# assignment 4.1.4
