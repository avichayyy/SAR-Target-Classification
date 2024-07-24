import os
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Define the path to the directory containing the class folders
base_path = r'C:\Users\Avichai\Documents\MATLAB\Data10c'

# Initialize lists to hold image data and labels
images = []
labels = []
class_names = []
include_classes = {'ZSU_23_4', 'BRDM2', '2S1', 'BRDM_2'}

# Helper function to recursively collect images and labels
def collect_images_from_folder(folder_path, label_suffix):
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.jpeg') or file_name.endswith('.JPG'):  # Add other image extensions if needed
                class_name = os.path.basename(root)  # Get the class name from the last folder
                if class_name not in include_classes:
                    continue
                if class_name not in class_names:
                    # Normalize class labels (combine BRDM2 and BRDM_2)
                    if class_name in ['BRDM2', 'BRDM_2']:
                        class_name = 'BRDM2'
                    class_names.append(class_name)

                file_path = os.path.join(root, file_name)
                image = Image.open(file_path).convert('L')  # Convert to grayscale
                image = image.resize((64, 64))  # Resize to a fixed size (64x64)
                images.append(np.array(image).flatten())
                labels.append(f"{class_name}_{label_suffix}")


# Collect images from 15_DEG and 17_DEG folders
collect_images_from_folder(os.path.join(base_path, '17_DEG'), '17')
collect_images_from_folder(os.path.join(base_path, '45_DEG'), '45')

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(images)

# Plot the results
plt.figure(figsize=(12, 8))
cmap = ListedColormap(plt.cm.get_cmap('tab10').colors[:len(class_names)])  # Use a colormap with enough colors

# Create a mapping from class names to colors
class_to_color = {class_name: cmap(i) for i, class_name in enumerate(class_names)}
markers = {'17': 'o', '45': 'x'}

for class_name in class_names:
    for angle in ['17', '45']:
        full_label = f"{class_name}_{angle}"
        indices = labels == full_label
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=full_label, alpha=0.6,
                    marker=markers[angle], color=class_to_color[class_name])

# Create legend with combined labels
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
plt.legend(unique_labels.values(), unique_labels.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

plt.title('t-SNE of Data - 17 vs 45 Degree')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
