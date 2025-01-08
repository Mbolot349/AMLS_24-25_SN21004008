import tensorflow as tf
from tensorflow.keras.models import model_from_json, load_model
import numpy as np
from tensorflow.keras.utils import to_categorical, normalize
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
# Path to the files

import os
folder = os.path.dirname(__file__)  # folder in which the script is located
keras_model_path = os.path.join(folder, 'task2.keras')
weights_path  = os.path.join(folder, 'weights_task2.weights.h5')



# Load the .keras file if available
try:
    model = load_model(keras_model_path)
    print("Model loaded successfully from task2.keras.")
except Exception as e:
    print("Could not load model from .keras file. Error:", e)
    print("Attempting to load model from JSON and weights.")

    # Load from JSON and weights if .keras fails
    

# Verify model summary
try:
    print("=========================model_summary======================================")
    model.summary()
except Exception as e:
    print("Could not display model summary. Error:", e)


from medmnist import BloodMNIST
test_data = BloodMNIST(split='test', download=True)

X_test, y_test = np.array(test_data.imgs), np.array(test_data.labels)
X_test, y_test = X_test / 255.0, (y_test)

label_map = {"0": "basophil",
            "1": "eosinophil",
            "2": "erythroblast",
            "3": "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)",
            "4": "lymphocyte",
            "5": "monocyte",
            "6": "neutrophil",
            "7": "platelet"}


X_test=normalize(X_test, axis=1)
y_test=to_categorical(y_test, num_classes=8)



A= model.evaluate(X_test,y_test)



y_pred=model.predict(X_test) 
y_pred1=np.argmax(y_pred, axis=1)
y_test1=np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test1, y_pred1)
print(cm)
#printing the Classificarion Report: 
print(classification_report(y_test1,y_pred1))

precision = precision_score(y_test1,y_pred1, average='weighted')
print('Precision: %f' % precision)
#recall:tp/(tp+fn)
recall = recall_score(y_test1,y_pred1, average='weighted')
print('Recall: %f' % recall)

f1 = f1_score(y_test1,y_pred1, average='weighted')
print('F1 score: %f' % f1)

print ('IoU:', jaccard_score(y_test1,y_pred1, average='micro'))

print("Accuracy_test:",accuracy_score(y_test1,y_pred1))


from sklearn.metrics import roc_auc_score

# Calculate the ROC-AUC score
roc_auc = roc_auc_score(y_test, y_pred)
print("ROC-AUC Score: %f" % roc_auc)