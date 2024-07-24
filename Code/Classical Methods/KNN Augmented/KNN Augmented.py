import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_score
import matplotlib.pyplot as plt
import seaborn as sns


import pickle

# File paths
x_train_path = r"C:\Users\Avichai\Desktop\Technion\temp\AConvNet-pytorch-main\src\dataset prepared\x_train.pkl"
y_train_path = r"C:\Users\Avichai\Desktop\Technion\temp\AConvNet-pytorch-main\src\dataset prepared\y_train.pkl"
x_test_path = r"C:\Users\Avichai\Desktop\Technion\temp\AConvNet-pytorch-main\src\dataset prepared\x_test.pkl"
y_test_path = r"C:\Users\Avichai\Desktop\Technion\temp\AConvNet-pytorch-main\src\dataset prepared\y_test.pkl"

# Load data
with open(x_train_path, 'rb') as file:
    X_train = pickle.load(file)

with open(y_train_path, 'rb') as file:
    y_train = pickle.load(file)

with open(x_test_path, 'rb') as file:
    X_test = pickle.load(file)

with open(y_test_path, 'rb') as file:
    y_test = pickle.load(file)

# Directory to save confusion matrices
output_folder = r'C:\Users\Avichai\Desktop\Technion\Project B\KNN Results\KNN - augmented'

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