from sklearn.datasets import make_blobs, make_gaussian_quantiles
from pdc_dp_means import DPMeans
import matplotlib.pyplot as plt


def plot_clusters(data, labels, title):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
    plt.title(title)
    plt.show()

X, y_true = make_blobs(n_samples=300, centers=2, cluster_std=0.60, random_state=0)
gaussian_data, gaussian_labels = make_gaussian_quantiles(n_samples=500, n_features=2, n_classes=2, random_state=0)

dpmeans = DPMeans(n_clusters=1,n_init=10, delta=15)  # n_init and delta parameters
dpmeans.fit(X)

y_dpmeans = dpmeans.predict(X)
gaussian_dpmeans = dpmeans.predict(gaussian_data)

plot_clusters(X, y_true, "Well-Separated Blobs and Correct Labels")
plot_clusters(X, y_dpmeans, "DP-Means on Well-Separated Blobs")
plot_clusters(gaussian_data, gaussian_labels, "Gaussian Quantiles and Correct Labels")
plot_clusters(gaussian_data, gaussian_dpmeans, "DP-Means on Gaussian Quantiles")
