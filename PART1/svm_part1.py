# Required Libraries
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
    roc_curve,
    auc,
)
from medmnist import BreastMNIST
import torch

# Load the Dataset
from medmnist import INFO, Evaluator
info = INFO['breastmnist']

# Load train, validation, and test splits from BreastMNIST
train_data = BreastMNIST(split='train', download=True)
val_data = BreastMNIST(split='val', download=True)
test_data = BreastMNIST(split='test', download=True)

# Convert datasets to numpy arrays
X_train, y_train = np.array(train_data.imgs), np.array(train_data.labels)
X_val, y_val = np.array(val_data.imgs), np.array(val_data.labels)
X_test, y_test = np.array(test_data.imgs), np.array(test_data.labels)

# Reshape input for visualization and expand dimensions
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Normalize the data
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

# Mapping numeric labels to text labels
label_map = {0: 'malignant', 1: 'benign'}

# Convert numeric labels to text labels for all datasets
y_train_text = np.array([label_map[label[0]] for label in y_train])
y_val_text = np.array([label_map[label[0]] for label in y_val])
y_test_text = np.array([label_map[label[0]] for label in y_test])

# Visualize some training images
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].squeeze(), cmap='gray')
    ax.set_title(f"Label: {y_train_text[i]}")
plt.tight_layout()
plt.show()

# Flatten the data for SVM
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Define the SVM model and hyperparameters
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [0.00001, 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly','linear']
}

svc = svm.SVC(probability=True)

# Perform grid search with cross-validation
grid_search = GridSearchCV(svc, param_grid, cv=3, verbose=2, n_jobs=-1)
grid_search.fit(X_train_flat, y_train.ravel())

# Best parameters
print(f"Best Parameters: {grid_search.best_params_}")

# Plotting Cross-Validation Results
results = pd.DataFrame(grid_search.cv_results_)
scores_mean = results['mean_test_score']
scores_std = results['std_test_score']
params = results['params']

# Extract C and gamma values
C_values = [param['C'] for param in params]
gamma_values = [param['gamma'] for param in params]
kernel_values = [param['kernel'] for param in params]

# Create a DataFrame for plotting
plot_data = pd.DataFrame({
    'mean_test_score': scores_mean,
    'C': C_values,
    'gamma': gamma_values,
    'kernel': kernel_values
})

# Plot mean test scores for different C and gamma values
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.pointplot(
    data=plot_data,
    x='C',
    y='mean_test_score',
    hue='gamma',
    palette='viridis',
    markers='o',
    linestyles='-'
)
plt.title('Cross-Validation Accuracy for Different Hyperparameters')
plt.ylabel('Mean Cross-Validation Accuracy')
plt.xlabel('C Values')
plt.legend(title='Gamma Values')
plt.show()

# Evaluate on the validation set
y_val_pred = grid_search.predict(X_val_flat)

# Map predicted labels to text labels for validation set
y_val_pred_text = np.array([label_map[label] for label in y_val_pred])

# Classification Report for Validation Set
print("Validation Report:")
print(classification_report(y_val_text, y_val_pred_text))

# Confusion Matrix for Validation Set
cm_val = confusion_matrix(y_val_text, y_val_pred_text, labels=['malignant', 'benign'])
disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=['malignant', 'benign'])
disp_val.plot()
plt.title("Confusion Matrix - Validation Data")
plt.show()

# Validation Accuracy
val_accuracy = accuracy_score(y_val_text, y_val_pred_text)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# ROC Curve and AUC for Validation Set
y_val_scores = grid_search.predict_proba(X_val_flat)[:, 1]  # Probability for the positive class ('benign')
fpr_val, tpr_val, thresholds_val = roc_curve(y_val.ravel(), y_val_scores, pos_label=1)
roc_auc_val = auc(fpr_val, tpr_val)

plt.figure()
plt.plot(fpr_val, tpr_val, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_val)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Validation Set')
plt.legend(loc="lower right")
plt.show()

# Predict on the test set
y_test_pred = grid_search.predict(X_test_flat)

# Map predicted labels to text labels for test set
y_test_pred_text = np.array([label_map[label] for label in y_test_pred])

# Classification Report for Test Set
print("Test Report:")
print(classification_report(y_test_text, y_test_pred_text))

# Confusion Matrix for Test Set
cm_test = confusion_matrix(y_test_text, y_test_pred_text, labels=['malignant', 'benign'])
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=['malignant', 'benign'])
disp_test.plot()
plt.title("Confusion Matrix - Test Data")
plt.show()

# Test Accuracy
test_accuracy = accuracy_score(y_test_text, y_test_pred_text)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# ROC Curve and AUC for Test Set
y_test_scores = grid_search.predict_proba(X_test_flat)[:, 1]  # Probability for the positive class ('benign')
fpr_test, tpr_test, thresholds_test = roc_curve(y_test.ravel(), y_test_scores, pos_label=1)
roc_auc_test = auc(fpr_test, tpr_test)

plt.figure()
plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_test)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Test Set')
plt.legend(loc="lower right")
plt.show()

# Load or input a new image
new_image = X_test[0]  # Example image from test set
plt.imshow(new_image.squeeze(), cmap='gray')
plt.title("New Image for Prediction")
plt.show()

# Preprocess and predict
new_image_flat = new_image.flatten().reshape(1, -1)
probability = grid_search.predict_proba(new_image_flat)
prediction = grid_search.predict(new_image_flat)

# Map the predicted label to text
prediction_text = label_map[prediction[0]]

print(f"Predicted Label: {prediction_text}")
print(f"Class Probabilities: {probability}")

# Print dataset sizes
print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
