import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles, make_gaussian_quantiles
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def plot_clusters(data, labels, title):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
    plt.title(title)
    plt.show()


blobs_data, blobs_labels = make_blobs(n_samples=300, cluster_std=0.60, centers=2, random_state=0)
moons_data, moons_labels = make_moons(n_samples=300, noise=0.1, random_state=0)
gaussian_data, gaussian_labels = make_gaussian_quantiles(n_samples=300, n_classes=2, random_state=0)

kmeans = KMeans(n_clusters=2, random_state=0)
blobs_kmeans_labels = kmeans.fit_predict(blobs_data)
moons_kmeans_labels = kmeans.fit_predict(moons_data)

plot_clusters(blobs_data, blobs_labels, "Well-Separated Blobs and Correct Labels")
plot_clusters(blobs_data, blobs_kmeans_labels, "K-Means on Well-Separated Blobs")

plot_clusters(moons_data, moons_labels, "Interlocking Moons and Correct Labels")
plot_clusters(moons_data, moons_kmeans_labels, "K-Means on Interlocking Moons")

gmm = GaussianMixture(n_components=2, random_state=0)
gaussian_gmm_labels = gmm.fit_predict(gaussian_data)
moons_gmm_labels = gmm.fit_predict(moons_data)

plot_clusters(moons_data, moons_labels, "Interlocking Moons and Correct Labels")
plot_clusters(moons_data, moons_gmm_labels, "GMM on Interlocking Moons")

plot_clusters(gaussian_data, gaussian_labels, "Gaussian Quantiles Labels")
plot_clusters(gaussian_data, gaussian_gmm_labels, "GMM on Gaussian Quantiles")
