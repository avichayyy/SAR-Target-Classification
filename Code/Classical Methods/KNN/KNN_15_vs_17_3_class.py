import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_score
import matplotlib.pyplot as plt
import seaborn as sns


# Function to load images from a directory and resize them, filtering by specific classes
def load_images_from_folder(folder, size=(128, 128), include_classes=None):
    images = []
    labels = []
    for class_name in os.listdir(folder):
        if include_classes and class_name not in include_classes:
            continue
        class_path = os.path.join(folder, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if filename.endswith('.jpg') or filename.endswith('.jpeg'):
                    img_path = os.path.join(class_path, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, size)
                        images.append(img.flatten())
                        labels.append(class_name)
    return np.array(images), np.array(labels)


# Load training and testing data
train_folder = r'C:\Users\Avichai\Documents\MATLAB\Data\17_DEG'
test_folder = r'C:\Users\Avichai\Documents\MATLAB\Data\15_DEG'
include_classes = {'ZSU_23_4', 'BRDM2', '2S1'}

X_train, y_train = load_images_from_folder(train_folder, include_classes=include_classes)
X_test, y_test = load_images_from_folder(test_folder, include_classes=include_classes)

# Directory to save confusion matrices
output_folder = r'C:\Users\Avichai\Desktop\Technion\Project B\KNN - 3 class'

# List of K values
k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35]

# Dictionary to store precision scores
precision_scores = {}

for k in k_values:
    # Train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict on the test set
    y_pred = knn.predict(X_test)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Save confusion matrix to file
    cm_file_path = os.path.join(output_folder, f'Confusion_matrix_K_{k}.png')
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for K={k}')
    plt.savefig(cm_file_path)
    plt.close()

    # Calculate and store precision
    precision = precision_score(y_test, y_pred, average='macro')
    precision_scores[k] = precision

# Print precision scores and find the highest precision
highest_precision = 0
best_k = None

for k, precision in precision_scores.items():
    print(f'Precision for K={k}: {precision:.4f}')
    if precision > highest_precision:
        highest_precision = precision
        best_k = k

print(f'Highest precision is for K={best_k}: {highest_precision:.4f}')
