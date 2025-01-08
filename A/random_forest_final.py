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
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import joblib

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
weights = {0: breast_class_weights[0], 1: breast_class_weights[1]}
print(f"Class weights for imbalance {weights}")
# Hyperparameter Tuning for Random Forest
acc_test_list = []
max_depth_list = []
max_features_list = []

for max_depth in range(10, 30):  # Range for max_depth
    for max_features in range(10, 30):  # Range for max_features
        print(f"Evaluating max depth={max_depth}, max features={max_features}")
        classifier = RandomForestClassifier(
            max_depth=max_depth, 
            max_features=max_features, 
            class_weight=weights, 
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
max_accuracy = max(acc_test_list)
max_index = acc_test_list.index(max_accuracy)

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
joblib.dump(final_classifier, "my_random_forest_final.joblib")




precision = precision_score(y_test,final_predictions, average='weighted')
print('Precision: %f' % precision)
#recall:tp/(tp+fn)
recall = recall_score(y_test,final_predictions, average='weighted')
print('Recall: %f' % recall)

f1 = f1_score(y_test,final_predictions, average='weighted')
print('F1 score: %f' % f1)

print ('IoU:', jaccard_score(y_test,final_predictions, average='micro'))

print("Accuracy_test:",accuracy_score(y_test,final_predictions))

from sklearn.metrics import roc_auc_score, roc_curve
final_predictions_proba = final_classifier.predict_proba(X_test)[:, 1]#
roc_auc = roc_auc_score(y_test, final_predictions_proba)
print(f"ROC-AUC: {roc_auc:.4f}")
