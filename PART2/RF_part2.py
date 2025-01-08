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
from medmnist import BloodMNIST
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import joblib
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight

# Load the Dataset
from medmnist import INFO
info = INFO['bloodmnist']
print(info)
# Load train, validation, and test splits from BreastMNIST
train_data = BloodMNIST(split='train', download=True)
val_data = BloodMNIST(split='val', download=True)
test_data = BloodMNIST(split='test', download=True)


# Convert datasets to numpy arrays
X_train, y_train = np.array(train_data.imgs), np.array(train_data.labels)
X_val, y_val = np.array(val_data.imgs), np.array(val_data.labels)
X_test, y_test = np.array(test_data.imgs), np.array(test_data.labels)
#




def convert_to_grayscale(images):
    # Sum the three channels and divide by 3 to get the grayscale image
    grayscale_images = images.mean(axis=-1)
    return grayscale_images

# Apply the conversion to train, validation, and test sets
X_train_gray = convert_to_grayscale(X_train)
X_val_gray = convert_to_grayscale(X_val)
X_test_gray = convert_to_grayscale(X_test)
print("Train set grayscale shape:", X_train_gray.shape)
print("Validation set grayscale shape:", X_val_gray.shape)
print("Test set grayscale shape:", X_test_gray.shape)
# Reshape (flatten) and normalize the data
print("normalizing")
X_train = X_train_gray.reshape(X_train.shape[0], -1) / 255.0
X_val = X_val_gray.reshape(X_val.shape[0], -1) / 255.0
X_test = X_test_gray.reshape(X_test.shape[0], -1) / 255.0
print("Train set grayscale shape:", X_train.shape)
print("Validation set grayscale shape:", X_val.shape)
print("Test set grayscale shape:", X_test.shape)
# Combine train and validation sets for better training
X_combined = np.concatenate((X_train, X_val), axis=0)
y_combined = np.concatenate((y_train, y_val), axis=0).flatten()




label_map = {"0": "basophil",
            "1": "eosinophil",
            "2": "erythroblast",
            "3": "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)",
            "4": "lymphocyte",
            "5": "monocyte",
            "6": "neutrophil",
            "7": "platelet"}
# Compute class weights for handling imbalance
blood_class_weights = class_weight.compute_class_weight(
    class_weight='balanced',  # Balanced strategy
    classes=np.unique(y_train[:, 0]),  # Extract unique classes
    y=y_train[:, 0]  # Labels for computing weights
)

# Convert the weights into a dictionary
weights = {i: blood_class_weights[i] for i in range(len(blood_class_weights))}

print(f"Class weights for imbalance: {weights}")

print("X Train: ", X_train.shape)
print("Y Train: ", y_train.shape)
print("X Test: ", X_test.shape)
print("Y Test: ", y_test.shape)
print("X test",X_test.shape)
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
# Use the full set of class labels
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[label_map[str(i)] for i in range(len(label_map))]
)
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

from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(
    y_test, 
    final_classifier.predict_proba(X_test), 
    multi_class='ovr'
)
print(f"ROC-AUC: {roc_auc:.4f}")