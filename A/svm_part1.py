import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from medmnist import BreastMNIST, INFO

info = INFO['breastmnist']

# Load train, validation, and test splits from BreastMNIST
train_data = BreastMNIST(split='train', download=True)
val_data = BreastMNIST(split='val', download=True)
test_data = BreastMNIST(split='test', download=True)

# Convert datasets to numpy arrays
X_train, y_train = np.array(train_data.imgs), np.array(train_data.labels)
X_val, y_val = np.array(val_data.imgs), np.array(val_data.labels)
X_test, y_test = np.array(test_data.imgs), np.array(test_data.labels)

# Reshape and normalize the data
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_val = X_val.reshape(X_val.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

# Combine train and validation sets for better training
X_combined = np.concatenate((X_train, X_val), axis=0)
y_combined = np.concatenate((y_train, y_val), axis=0).flatten()
print("shape of training data, images:",X_combined.shape)

print("shape of training data, labels:",y_combined.shape)
# Mapping numeric labels to text labels
label_map = {0: 'malignant', 1: 'benign/normal'}

# Convert numeric labels to text labels for all datasets
y_train_text = np.array([label_map[label[0]] for label in y_train])
y_val_text = np.array([label_map[label[0]] for label in y_val])
y_test_text = np.array([label_map[label[0]] for label in y_test])

# Visualize some training images
# Reshape the flattened images back to 28x28 for visualization
#fig, axes = plt.subplots(1, 5, figsize=(15, 3))
#for i, ax in enumerate(axes.flat):
#    ax.imshow(X_train[i].reshape(28, 28), cmap='gray')  # Reshape to 28x28
#    ax.set_title(f"Label: {y_train_text[i]}")
#    ax.axis('off')  # Turn off the axis for better visualization
#plt.tight_layout()
#plt.show()
#computes class weights
breast_class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_combined),
    y=y_combined
)
weights = {0: breast_class_weights[0], 1: breast_class_weights[1]}
print(f"Class weights for imbalance {weights}")
# Define the SVM model and hyperparameters
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [0.00001, 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly','linear']
}


svc = svm.SVC(probability=True,class_weight=weights)

# Perform grid search with cross-validation
grid_search = GridSearchCV(svc, param_grid, cv=3, verbose=2, n_jobs=-1)
grid_search.fit(X_combined, y_combined.ravel())

# Best parameters
print(f"Best Parameters: {grid_search.best_params_}")

# Evaluate on the test set
y_test_pred = grid_search.predict(X_test)

# Map predicted labels to text labels for validation set
y_test_pred_text = np.array([label_map[label] for label in y_test_pred])

# Classification Report for Test Set
print("Validation Report:")
print(classification_report(y_test_text, y_test_pred_text))

# Confusion Matrix for test Set
cm_val = confusion_matrix(y_test_text, y_test_pred_text, labels=['malignant', 'benign/normal'])
disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=['malignant', 'benign/normal'])
disp_val.plot()
plt.title("Confusion Matrix - Validation Data")
plt.show()

# test Accuracy
val_accuracy = accuracy_score(y_test_text, y_test_pred_text)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")




precision = precision_score(y_test,y_test_pred, average='weighted')
print('Precision: %f' % precision)
#recall: tp/(tp+fn)
recall = recall_score(y_test,y_test_pred, average='weighted')
print('Recall: %f' % recall)

f1 = f1_score(y_test,y_test_pred, average='weighted')
print('F1 score: %f' % f1)

print ('IoU:', jaccard_score(y_test,y_test_pred, average='micro'))

print("Accuracy_test:",accuracy_score(y_test,y_test_pred))

from sklearn.metrics import roc_auc_score, roc_curve
# Get the probabilities for the positive class (class 1)
y_test_pred_proba = grid_search.predict_proba(X_test)[:, 1]

# Calculate the ROC-AUC
roc_auc = roc_auc_score(y_test, y_test_pred_proba)
print(f"ROC-AUC: {roc_auc:.4f}")
