import tensorflow as tf
from tensorflow.keras.models import model_from_json, load_model
import numpy as np
import matplotlib.pyplot as plt
# Path to the files
import os
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
folder = os.path.dirname(__file__)  # folder in which the script is located
keras_model_path = os.path.join(folder, 'task1.keras')
weights_path  = os.path.join(folder, 'weights_task1.weights.h5')




# Load the .keras file if available
try:
    model = load_model(keras_model_path)
    print("Model loaded successfully from task1.keras.")
except Exception as e:
    print("Could not load model from .keras file. Error:", e)
    print("Attempting to load model from JSON and weights.")

    # Load from JSON and weights if .keras fails
    try:
        with open(json_model_path, 'r') as json_file:
            json_model = json_file.read()
        model = model_from_json(json_model)
        model.load_weights(weights_path)
        print("Model loaded successfully from Task1_CNN.json and weights_task1.weights.h5.")
    except Exception as e:
        print("Failed to load model from JSON and weights. Error:", e)

# Verify model summary
try:
    print("===========================model summary======================================")
    model.summary()
except Exception as e:
    print("Could not display model summary. Error:", e)

print("--------------------------------Testing on BreastMNIST Test DATASET-------------------------------------")
from medmnist import BreastMNIST
test_data = BreastMNIST(split='test', download=True)

X_test, y_test = np.array(test_data.imgs), np.array(test_data.labels)


X_test = X_test[..., np.newaxis] / 255.0 


label_map = {0: 'malignant', 1: 'benign/normal'}
y_test_text = np.array([label_map[label[0]] for label in y_test])
from sklearn.metrics import roc_auc_score
y_pred=model.predict(X_test) 
#y_pred1
y_pred1= y_pred.round()

cm = confusion_matrix(y_test, y_pred1)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_map.values()))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print(classification_report(y_test,y_pred1))

precision = precision_score(y_test,y_pred1, average='weighted')
print('Precision: %f' % precision)
#recall: tp/(tp+fn)
recall = recall_score(y_test,y_pred1, average='weighted')
print('Recall: %f' % recall)
#f1: tp/(tp+fp+fn)
f1 = f1_score(y_test,y_pred1, average='weighted')
print('F1 score: %f' % f1)

print ('IoU:', jaccard_score(y_test,y_pred1, average='micro'))

print("Accuracy_test:",accuracy_score(y_test,y_pred1))


from sklearn.metrics import roc_auc_score

# Calculate the ROC-AUC score
roc_auc = roc_auc_score(y_test, y_pred)
print("ROC-AUC Score: %f" % roc_auc)
