import os
import numpy as np
from PIL import Image
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Define the path to the directory containing the class folders
base_path = r'C:\Users\Avichai\Documents\MATLAB\Data'

# Initialize lists to hold image data and labels
images = []
labels = []
class_names_set = set()

# Read images from each class folder
for degree_folder in os.listdir(base_path):
    degree_path = os.path.join(base_path, degree_folder)
    if os.path.isdir(degree_path):
        for root, _, files in os.walk(degree_path):
            for file_name in files:
                if file_name.lower().endswith(('.jpg', '.jpeg')):  # Add other image extensions if needed
                    file_path = os.path.join(root, file_name)
                    class_name = os.path.basename(os.path.dirname(file_path))

                    # Normalize class labels (combine BRDM2 and BRDM_2)
                    if class_name in ['BRDM2', 'BRDM_2']:
                        class_name = 'BRDM2'

                    class_names_set.add(class_name)
                    image = Image.open(file_path).convert('L')  # Convert to grayscale
                    image = image.resize((64, 64))  # Resize to a fixed size (64x64)
                    images.append(np.array(image).flatten())
                    labels.append(class_name)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)
class_names = list(class_names_set)

# Perform Laplacian Eigenmaps
lem = SpectralEmbedding(n_components=2, random_state=42)
lem_results = lem.fit_transform(images)

# Plot the results
plt.figure(figsize=(12, 8))
cmap = ListedColormap(plt.cm.get_cmap('tab10').colors[:len(class_names)])  # Use a colormap with enough colors

for class_name in class_names:
    indices = labels == class_name
    plt.scatter(lem_results[indices, 0], lem_results[indices, 1], label=class_name, alpha=0.6)

plt.legend()
plt.title('Laplacian Eigenmaps (LEM) of Image Data')
plt.xlabel('LEM Component 1')
plt.ylabel('LEM Component 2')
plt.show()
