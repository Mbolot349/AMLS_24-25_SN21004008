import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from medmnist import BreastMNIST
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



test_data = BreastMNIST(split='test', download=True)

# Convert datasets to numpy arrays
X_test, y_test = np.array(test_data.imgs), np.array(test_data.labels)

# Reshape (flatten) and normalize the data
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

import os
folder = os.path.dirname(__file__)  # folder in which the script is located
model_path = os.path.join(folder, 'my_random_forest_part1.joblib')


loaded_model = joblib.load(model_path)
print("model has been loaded")


final_predictions = loaded_model.predict(X_test)
final_accuracy = accuracy_score(y_test.flatten(), final_predictions)
print("Final Model Evaluation:")
print(classification_report(y_test.flatten(), final_predictions))
print(f"Final Accuracy: {final_accuracy}")

# Confusion Matrix
cm = confusion_matrix(y_test.flatten(), final_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['malignant', 'benign/normal'])
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