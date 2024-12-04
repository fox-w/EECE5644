import cv2
import numpy as np

# Load the image
image = cv2.imread("42049.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Downsample image if necessary
image = cv2.resize(image, (200, 200))  # Example downsample

# --------------------------------

# Get image dimensions
rows, cols, _ = image.shape

# Create row and column indices
row_indices = np.arange(rows).repeat(cols).reshape(rows, cols)
col_indices = np.arange(cols).reshape(1, -1).repeat(rows, axis=0)

# Stack features
features = np.stack([
    row_indices.flatten() / rows,  # Row index normalized
    col_indices.flatten() / cols,  # Column index normalized
    image[:, :, 0].flatten() / 255.0,  # Red normalized
    image[:, :, 1].flatten() / 255.0,  # Green normalized
    image[:, :, 2].flatten() / 255.0   # Blue normalized
], axis=1)

# --------------------------------

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold

# Cross-validation for model order selection
best_log_likelihood = -np.inf
best_model = None
n_components_range = range(2, 11)  # Number of clusters to test

for n_components in n_components_range:
    avg_log_likelihood = 0
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(features):
        train_features, val_features = features[train_idx], features[val_idx]
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(train_features)
        avg_log_likelihood += gmm.score(val_features) / kf.get_n_splits()
    
    if avg_log_likelihood > best_log_likelihood:
        best_log_likelihood = avg_log_likelihood
        best_model = gmm

print(f"Best number of components: {best_model.n_components}")

# --------------------------------

# Assign labels to each pixel
labels = best_model.predict(features)

# Reshape labels to image dimensions
label_image = labels.reshape(rows, cols)

# Normalize labels for visualization (map to grayscale)
label_image_normalized = (255 * (label_image / label_image.max())).astype(np.uint8)

# --------------------------------
# Plot images
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

# Segmented image
plt.subplot(1, 2, 2)
plt.imshow(label_image_normalized, cmap="gray")
plt.title("Segmented Image")
plt.axis("off")

plt.tight_layout()
plt.show()
