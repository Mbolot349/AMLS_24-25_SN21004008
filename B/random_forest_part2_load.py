import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from medmnist import BloodMNIST
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
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
import os
folder = os.path.dirname(__file__)  # folder in which the script is located
model_path = os.path.join(folder, 'my_random_forest_part2.joblib')
# Path to your saved model 

loaded_model = joblib.load(model_path)
print("model has been loaded")
test_data = BloodMNIST(split='test', download=True)
X_test, y_test = np.array(test_data.imgs), np.array(test_data.labels)

def convert_to_grayscale(images):
    # Sum the three channels and divide by 3 to get the grayscale image
    grayscale_images = images.mean(axis=-1)
    return grayscale_images

# Apply the conversion to train, validation, and test sets
X_test_gray = convert_to_grayscale(X_test)
X_test = X_test_gray.reshape(X_test.shape[0], -1) / 255.0

label_map = {"0": "basophil",
            "1": "eosinophil",
            "2": "erythroblast",
            "3": "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)",
            "4": "lymphocyte",
            "5": "monocyte",
            "6": "neutrophil",
            "7": "platelet"}

final_predictions = loaded_model.predict(X_test)

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



precision = precision_score(y_test,final_predictions, average='weighted')
print('Precision: %f' % precision)
#recall:tp/(tp+fn)
recall = recall_score(y_test,final_predictions, average='weighted')
print('Recall: %f' % recall)

f1 = f1_score(y_test,final_predictions, average='weighted')
print('F1 score: %f' % f1)

print ('IoU:', jaccard_score(y_test,final_predictions, average='micro'))

print("Accuracy_test:",accuracy_score(y_test,final_predictions))