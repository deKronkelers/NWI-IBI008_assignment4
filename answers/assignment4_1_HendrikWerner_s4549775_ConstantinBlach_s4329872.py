# author: Hendrik Werner s4549775
# author: Constantin Blach s4329872
from random import randrange

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.spatial import distance
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
X = loadmat("./data/wildfaces.mat")["X"]
k = 10
centroids, labels, inertia = k_means(X, k)


def find_min_distance_centroid(element, centroids) -> int:
    min_index = 0
    element = np.reshape(element, centroids[0].shape)
    min_distance = distance.euclidean(element, centroids[0])
    for i in range(1, centroids.shape[0]):
        dist = distance.euclidean(element, centroids[i])
        if dist < min_distance:
            min_distance = dist
            min_index = i
    return min_index


def plot_face_against_centroid(centroids, index: int):
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(np.reshape(X[index, :], (3, 40, 40)).T)
    ax[0].set_title("Face {}".format(index))
    min_index = find_min_distance_centroid(X[index, :], centroids)
    ax[1].imshow(np.reshape(centroids[min_index], (3, 40, 40)).T)
    ax[1].set_title("Centroid {}".format(min_index))
    plt.show()


for i in range(3):
    plot_face_against_centroid(centroids, randrange(X.shape[0]), k)

# assignment 4.1.4
