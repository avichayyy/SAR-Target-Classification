import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Define the path to the directory containing the class folders
base_path = r'C:\Users\Avichai\Documents\MATLAB\Data10c'

# Initialize lists to hold image data and labels
images = []
labels = []
class_names = []


# Helper function to recursively collect images and labels
def collect_images_from_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.jpeg'):  # Add other image extensions if needed
                class_name = os.path.basename(root)  # Get the class name from the last folder
                if class_name not in class_names:
                    # Normalize class labels (combine BRDM2 and BRDM_2)
                    if class_name in ['BRDM2', 'BRDM_2']:
                        class_name = 'BRDM2'
                    class_names.append(class_name)

                file_path = os.path.join(root, file_name)
                image = Image.open(file_path).convert('L')  # Convert to grayscale
                image = image.resize((64, 64))  # Resize to a fixed size (64x64)
                images.append(np.array(image).flatten())
                labels.append(f"{class_name}")


base_path = r'C:\Users\Avichai\Documents\MATLAB\Data10c'
collect_images_from_folder(base_path)
# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)


# Function to perform PCA and plot the results
def perform_pca_and_plot(data, title):
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(data)

    plt.figure(figsize=(12, 8))
    cmap = ListedColormap(plt.cm.get_cmap('tab10').colors[:len(class_names)])  # Use a colormap with enough colors

    for class_name in class_names:
        indices = labels == class_name
        plt.scatter(pca_results[indices, 0], pca_results[indices, 1], label=class_name, alpha=0.6)

    plt.legend()
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()


# Perform PCA on non-normalized data
perform_pca_and_plot(images, 'PCA of Non-Normalized Image Data')

# Normalize the data
scaler = StandardScaler()
images_normalized = scaler.fit_transform(images)

# Perform PCA on normalized data
perform_pca_and_plot(images_normalized, 'PCA of Normalized Image Data')
