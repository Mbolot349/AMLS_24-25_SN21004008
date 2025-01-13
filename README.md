# ELEC0134-AMLS_24-25_SN21004008
## Abstract
This repository contains the code for the ELEC0134 Coursework (2024-2025). The project is divided into two parts.

** Part 1 - BreastMNIST **
The first task is a binary classification task focusing on identifying malignant or bening/normal tissues from ultrasound images of breast tissue. The dataset contains 780 greyscale images split into a ratio of 7 : 1 : 2 of training : validation : testing with labels 0: malignant, 1: normal/benign. Folder "A" contains the code for training the models. The models compared for this task are Support Vector Machine (SVM), Random Forest (RF) and convolutional neural network CNN. 

** Part 2 - BloodMNIST **
The second task is a multi-class classification task focusing on identifying different types of blood cell types. The dataset contains 17,092 red-green-blue images split into a ratio of 7 : 1 : 2 of training : validation : testing with labels 0: basophil, 1: eosinophil, 2: erythroblast, 3: immature granulocytes(myelocytes, metamyelocytes and promyelo-cytes), 4: lymphocyte, 5: monocyte, 6: neutrophil, 7: plate-let. Folder "B" contains the code for training the models. The models compared for this task are RF model and CNN model. 

The file of this reporitory is organised as follows:
* AMLS_24-25_SN21004008:
  *README.md
  *A:
    *svm_part1.py : Python file used to train the SVM model
    *random_forest_final.py : Python file used to train RF model
    *neural_final.py : Python file used to train CNN model
    *my_random_forest_part1.joblib : Saved RF model in joblib format
    *random_forest_part1_load.py : Python file used to load and test the RF model saved (to save time as execution time can be long)
    *task1.keras : Keras model file of the CNN model
    *weights_task1.weights.h5 : h5 file containg model weights for CNN in part 1
    *load_CNN_part1.py : Python file to load and test the CNN model trained in neural_final.py
