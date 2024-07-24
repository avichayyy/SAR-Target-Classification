import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# Function to load images from a directory and resize them
def load_images_from_folder(folder, size=(128, 128), exclude_class="SLICY"):
    images = []
    labels = []
    for class_name in os.listdir(folder):
        if class_name == exclude_class:
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

X_train, y_train = load_images_from_folder(train_folder)
X_test, y_test = load_images_from_folder(test_folder)

# Directory to save confusion matrices
output_folder = r'C:\Users\Avichai\Desktop\Technion\Project B\RandomForest'

# Initialize RandomForest classifier
rf = RandomForestClassifier(n_estimators=100)

# Train RandomForest classifier
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Print classification report
print(classification_report(y_test, y_pred))

# Save the model
model_file_path = os.path.join(output_folder, 'RandomForest_model.joblib')
joblib.dump(rf, model_file_path)

# Print model size
model_size = os.path.getsize(model_file_path) / (1024 * 1024)  # Size in MB
print(f'Model size: {model_size:.2f} MB')
