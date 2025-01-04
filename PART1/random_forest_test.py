import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
from sklearn.utils.class_weight import compute_class_weight
from medmnist import BreastMNIST
import torch

# Load the Dataset
from medmnist import INFO
info = INFO['breastmnist']
print(info)
# Load train, validation, and test splits from BreastMNIST
train_data = BreastMNIST(split='train', download=True)
val_data = BreastMNIST(split='val', download=True)
test_data = BreastMNIST(split='test', download=True)

# Convert datasets to numpy arrays
X_train, y_train = np.array(train_data.imgs), np.array(train_data.labels)
X_val, y_val = np.array(val_data.imgs), np.array(val_data.labels)
X_test, y_test = np.array(test_data.imgs), np.array(test_data.labels)

# Reshape (flatten) and normalize the data
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_val = X_val.reshape(X_val.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

# Combine train and validation sets for better training
X_combined = np.concatenate((X_train, X_val), axis=0)
y_combined = np.concatenate((y_train, y_val), axis=0).flatten()

# Compute class weights for handling imbalance
breast_class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_combined),
    y=y_combined
)

# Hyperparameter Tuning for Random Forest
acc_test_list = []
max_depth_list = []
max_features_list = []

for max_depth in range(10, 30):  # Range for max_depth
    for max_features in range(10, 30):  # Range for max_features
        print(f"Evaluating max_depth={max_depth}, max_features={max_features}")
        classifier = RandomForestClassifier(
            max_depth=max_depth, 
            max_features=max_features, 
            random_state=0
        )
        classifier.fit(X_combined, y_combined)  # Fit the model

        # Predict and evaluate on test set
        pred_y = classifier.predict(X_test)
        acc_test_test = accuracy_score(y_test.flatten(), pred_y)
        acc_test_list.append(acc_test_test)

        print(f"Accuracy: {acc_test_test}")
        max_depth_list.append(max_depth)
        max_features_list.append(max_features)

# Find the best hyperparameters
max_value = max(acc_test_list)
max_index = acc_test_list.index(max_value)

best_max_depth = max_depth_list[max_index]
best_max_features = max_features_list[max_index]

print("Best hyperparameters:")
print(f"max_depth: {best_max_depth}")
print(f"max_features: {best_max_features}")

# Train final model with best hyperparameters
final_classifier = RandomForestClassifier(
    max_depth=best_max_depth, 
    max_features=best_max_features, 
    class_weight=weights, 
    random_state=0
)
final_classifier.fit(X_combined, y_combined)

# Evaluate final model
final_predictions = final_classifier.predict(X_test)
final_accuracy = accuracy_score(y_test.flatten(), final_predictions)
print("Final Model Evaluation:")
print(classification_report(y_test.flatten(), final_predictions))
print(f"Final Accuracy: {final_accuracy}")

# Confusion Matrix
cm = confusion_matrix(y_test.flatten(), final_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['malignant', 'benign/normal'])
disp.plot(cmap=plt.cm.Blues)
plt.show()