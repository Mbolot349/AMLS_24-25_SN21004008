import tensorflow as tf
from tensorflow.keras.models import model_from_json, load_model
import numpy as np
# Path to the files
keras_model_path = "task1.keras"
json_model_path = "Task1_CNN.json"
weights_path = "weights_task1.weights.h5"

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
    model.summary()
except Exception as e:
    print("Could not display model summary. Error:", e)


from medmnist import BreastMNIST
test_data = BreastMNIST(split='test', download=True)

X_test, y_test = np.array(test_data.imgs), np.array(test_data.labels)


X_test = X_test[..., np.newaxis] / 255.0 


label_map = {0: 'malignant', 1: 'benign/normal'}
y_test_text = np.array([label_map[label[0]] for label in y_test])
from sklearn.metrics import roc_auc_score
y_pred=model.predict(X_test) 
roc_auc = roc_auc_score(y_test, y_pred, average='weighted', multi_class='ovr')
print("ROC-AUC Score: %f" % roc_auc)