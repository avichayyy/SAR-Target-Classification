import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from src import model
from src.data import mstar
from src.train import load_dataset
import sys

sys.path.append('../src')

def load_model(model_path, classes, channels):
    m = model.Model(classes=classes, channels=channels)
    m.load(model_path)
    return m


def select_vote(a, b, c, d):
    # Concatenate tensors into a single tensor
    numbers = torch.stack([a, b, c, d])

    # Count occurrences of each number
    counts = torch.bincount(numbers)

    # Find the number with maximum votes
    max_count = torch.max(counts)
    candidate = torch.nonzero(counts == max_count)[0].item()

    return candidate

def majority_vote(predictions):
    a = predictions[0]
    for i in range(len(predictions[0])):
        a[i] = select_vote(predictions[0][i],predictions[1][i],predictions[2][i],predictions[3][i])
    return a

def confusion_matrix(_m, ds):
    _pred = []
    _gt = []
    for m in _m:
        m.net.eval()
    _softmax = torch.nn.Softmax(dim=1)
    for i, data in enumerate(ds):
        images, labels, _ = data
        predictions_list = []
        for m in _m:
            predictions = m.inference(images)
            predictions = _softmax(predictions)
            _, predictions = torch.max(predictions.data, 1)
            predictions_list.append(predictions)

        predictions = majority_vote(predictions_list)
        labels = labels.type(torch.LongTensor)

        _pred += predictions.cpu().tolist()
        _gt += labels.cpu().tolist()

    conf_mat = metrics.confusion_matrix(_gt, _pred)
    accuracy = metrics.accuracy_score(_gt, _pred)
    
    return conf_mat, accuracy


# Paths to the model files
model_paths = [
    r"/root/AConvNet-pytorch-main/AConvNet-pytorch-main/ensemble/model-034.pth",
    #r"/root/AConvNet-pytorch-main/AConvNet-pytorch-main/ensemble/model-069.pth",
    r"/root/AConvNet-pytorch-main/AConvNet-pytorch-main/ensemble/model-036.pth",
    #r"/root/AConvNet-pytorch-main/AConvNet-pytorch-main/ensemble/model-027.pth",
    r"/root/AConvNet-pytorch-main/AConvNet-pytorch-main/ensemble/model-015.pth",
    #r"/root/AConvNet-pytorch-main/AConvNet-pytorch-main/ensemble/model-022.pth",
    r"/root/AConvNet-pytorch-main/AConvNet-pytorch-main/ensemble/model-017.pth",
]

# Load models
models = [load_model(path, classes=4, channels=2) for path in model_paths]

# Load dataset
test_set = load_dataset('dataset', False, 'eoc-1-t72-a64', 100)

# Calculate confusion matrix and accuracy
_conf_mat, accuracy = confusion_matrix(models, test_set)

# Plot confusion matrix
sns.reset_defaults()
ax = sns.heatmap(_conf_mat, annot=True, fmt='d', cbar=False)
ax.set_yticklabels(mstar.target_name_eoc_1, rotation=0)
ax.set_xticklabels(mstar.target_name_eoc_1, rotation=30)

plt.xlabel('prediction', fontsize=12)
plt.ylabel('label', fontsize=12)
plt.savefig('confusion_matrix.png')

plt.close()

print(f"Accuracy: {accuracy*100:.2f}")
