import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FastICA


def generate_bad_pca_data(n_points=100):
    phi = np.random.uniform(0, np.pi, n_points)
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    r = 1
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.column_stack((x, y, z))


def generate_good_lda_data(n_points=100):
    x1 = np.random.multivariate_normal([2, 2, 2], np.eye(3), n_points)
    x2 = np.random.multivariate_normal([-2, -2, -2], np.eye(3), n_points)
    y1 = np.zeros(n_points)
    y2 = np.ones(n_points)
    X = np.vstack((x1, x2))
    y = np.hstack((y1, y2))
    return X, y


def generate_overlapping_blobs(n_samples, centers, covariances):
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=1, random_state=0)
    for i in range(len(centers)):
        X[y == i] = np.dot(X[y == i], covariances[i])
    return X, y


centers = [[0, 0, 0], [2, 2, 2], [-2, -2, -2]]
covariances = [
    [[0.8, 0.3, 0.1], [0.3, 0.7, 0.2], [0.1, 0.2, 0.6]],
    [[0.5, -0.2, 0], [-0.2, 0.7, 0.1], [0, 0.1, 0.4]],
    [[0.9, 0, 0], [0, 0.3, 0], [0, 0, 0.2]]
]


def generate_linear_data(n_points=150):
    x = np.random.rand(n_points)
    y1 = 2 * x + np.random.normal(0, 0.05, n_points)
    y2 = 3 * x + np.random.normal(0, 0.05, n_points)
    X = np.column_stack((x, y1, y2))
    Z = np.column_stack((y1, y2, x))
    return X, Z


def generate_bad_cca_data(n_points=150):
    X = np.random.rand(n_points, 3)
    Z = np.random.rand(n_points, 3)
    return X, Z


def generate_good_ica_data(n_points=100):
    S = np.random.laplace(size=(n_points, 3))
    A = np.array([[1, 1, 1], [0.5, 2, 1.5], [1.5, 1, 2]])
    X = np.dot(S, A.T)
    return X


def generate_bad_ica_data(n_points=100):
    X = np.random.normal(size=(n_points, 3))
    return X


def plot_3d(X, y, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
    ax.set_title(title)
    plt.show()


def plot_2d(X, y, title):
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title(title)
    plt.show()


def plot_all_four(good_data, good_transformed, bad_data, bad_transformed, algo_name):
    fig = plt.figure(figsize=(12, 6))

    ax = fig.add_subplot(231, projection='3d')
    ax.scatter(good_data[:, 0], good_data[:, 1], good_data[:, 2])
    ax.set_title(f'Good {algo_name} Data - 3D Before {algo_name}')

    ax = fig.add_subplot(232)
    ax.scatter(good_transformed[:, 0], good_transformed[:, 1])
    ax.set_title(f'Good {algo_name} Data - 2D After {algo_name}')

    ax = fig.add_subplot(234, projection='3d')
    ax.scatter(bad_data[:, 0], bad_data[:, 1], bad_data[:, 2])
    ax.set_title(f'Bad {algo_name} Data - 3D Before {algo_name}')

    ax = fig.add_subplot(235)
    ax.scatter(bad_transformed[:, 0], bad_transformed[:, 1])
    ax.set_title(f'Bad {algo_name} Data - 2D After {algo_name}')

    plt.tight_layout()
    plt.show()


# Extract data
good_pca_data, _ = generate_linear_data()
bad_pca_data = generate_bad_pca_data()
good_lda_data, good_lda_labels = make_blobs(n_samples=1000, centers=3, n_features=3, random_state=0)
bad_lda_data, bad_lda_labels = generate_overlapping_blobs(1000, centers, covariances)
good_cca_data, good_cca_data_Z = generate_linear_data()
bad_cca_data, bad_cca_data_Z = generate_bad_cca_data()
good_ica_data = generate_good_ica_data()
bad_ica_data = generate_bad_ica_data()

# Initialize algorithms
pca = PCA(n_components=2)
lda = LDA(n_components=2)
cca = CCA(n_components=2)
ica = FastICA(n_components=2)

# Transform
good_pca_transformed = pca.fit_transform(good_pca_data)
bad_pca_transformed = pca.fit_transform(bad_pca_data)
good_lda_transformed = lda.fit_transform(good_lda_data, good_lda_labels)
bad_lda_transformed = lda.fit_transform(bad_lda_data, bad_lda_labels)
good_cca_transformed, good_cca_transformed_Z = cca.fit_transform(good_cca_data, good_cca_data_Z)
bad_cca_transformed, bad_cca_transformed_Z = cca.fit_transform(bad_cca_data, bad_cca_data_Z)
good_ica_transformed = ica.fit_transform(good_ica_data)
bad_ica_transformed = ica.fit_transform(bad_ica_data)

# Plot
plot_all_four(good_pca_data, good_pca_transformed, bad_pca_data, bad_pca_transformed, "PCA")
plot_3d(good_lda_data, good_lda_labels, '3D Blobs')
plot_2d(good_lda_transformed, good_lda_labels, 'LDA transformed Blobs')
plot_3d(bad_lda_data, bad_lda_labels, 'Overlapped Blobs')
plot_2d(bad_lda_transformed, bad_lda_labels, 'LDA transformed Overlapped Blobs')
plot_all_four(good_cca_data, good_cca_transformed, bad_cca_data, bad_cca_transformed, "CCA")
plot_all_four(good_ica_data, good_ica_transformed, bad_ica_data, bad_ica_transformed, "ICA")
