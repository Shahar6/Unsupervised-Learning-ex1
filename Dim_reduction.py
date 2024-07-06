import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def generate_good_pca_data(n_points=100):
    # Create data with a linear structure
    x = np.random.rand(n_points)
    y = 2 * x + np.random.normal(0, 0.1, n_points)
    z = 3 * x + np.random.normal(0, 0.1, n_points)
    return np.column_stack((x, y, z))


def generate_bad_pca_data(n_points=100):
    # Create data with a spherical structure
    phi = np.random.uniform(0, np.pi, n_points)
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    r = 1
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.column_stack((x, y, z))


good_pca_data = generate_good_pca_data()
bad_pca_data = generate_bad_pca_data()

pca = PCA(n_components=2)

good_pca_transformed = pca.fit_transform(good_pca_data)
bad_pca_transformed = pca.fit_transform(bad_pca_data)

# Plot the original 3D data and the PCA results
fig = plt.figure(figsize=(12, 6))

# Plot good PCA data
ax = fig.add_subplot(231, projection='3d')
ax.scatter(good_pca_data[:, 0], good_pca_data[:, 1], good_pca_data[:, 2])
ax.set_title('Good PCA Data - 3D Before PCA')

ax = fig.add_subplot(232)
ax.scatter(good_pca_transformed[:, 0], good_pca_transformed[:, 1])
ax.set_title('Good PCA Data - 2D After PCA')

# Plot bad PCA data
ax = fig.add_subplot(234, projection='3d')
ax.scatter(bad_pca_data[:, 0], bad_pca_data[:, 1], bad_pca_data[:, 2])
ax.set_title('Bad PCA Data - 3D Before PCA')

ax = fig.add_subplot(235)
ax.scatter(bad_pca_transformed[:, 0], bad_pca_transformed[:, 1])
ax.set_title('Bad PCA Data - 2D After PCA')

plt.tight_layout()
plt.show()
