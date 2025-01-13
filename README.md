# ELEC0134-AMLS_24-25_SN21004008
## Abstract
This repository contains the code for the ELEC0134 Coursework (2024-2025). The project is divided into two parts.

**Part 1 - BreastMNIST**:

The first task is a binary classification task focusing on identifying malignant or bening/normal tissues from ultrasound images of breast tissue. The dataset contains 780 greyscale images split into a ratio of 7 : 1 : 2 of training : validation : testing with labels 0: malignant, 1: normal/benign. Folder "A" contains the code for training the models. The models compared for this task are Support Vector Machine (SVM), Random Forest (RF) and convolutional neural network CNN. 

**Part 2 - BloodMNIST**:

The second task is a multi-class classification task focusing on identifying different types of blood cell types. The dataset contains 17,092 red-green-blue images split into a ratio of 7 : 1 : 2 of training : validation : testing with labels 0: basophil, 1: eosinophil, 2: erythroblast, 3: immature granulocytes(myelocytes, metamyelocytes and promyelo-cytes), 4: lymphocyte, 5: monocyte, 6: neutrophil, 7: plate-let. Folder "B" contains the code for training the models. The models compared for this task are RF model and CNN model. 

The file of this repository is organised as follow:

- AMLS_24-25_SN21004008:

  - README.md: Contains some information about the repository.

  - "A" Folder: This folder contains all the files for Task 1, which are:
    
    ± svm_part1.py: This Python file is used to train the SVM model.
    
    ± random_forest_final.py: This Python file is used to train the RF model.
    
    ± neural_final.py: This Python file is used to train the CNN model.
    
    ± my_random_forest_part1.joblib: Saved RF model in joblib format.
    
    ± random_forest_part1_load.py: This Python file is used to load and test the RF model (to save time as execution time can be long).
    
    ± task1.keras: Keras model file for the CNN model.
    
    ± weights_task1.weights.h5: h5 file containing model weights for the CNN in part 1.
    
    ± load_CNN_part1.py: This Python file is used to load and test the CNN model trained in neural_final.py.

  - "B" Folder: This folder contains all the files for Task 2, which are:
    
     ± RF_part2.py: This Python file is used to train the RF model.

     ±  part2_CNN.py: This Python file is used to train the CNN model.

    ±  my_random_forest_part2.joblib: Saved RF model in joblib format.

    ± random_forest_part2_load.py: This Python file is used to load and test the RF model (to save time as execution time can be long).

    ± task2.keras: Keras model file for the CNN model.
    
    ± weights_task2.weights.h5: h5 file containing model weights for the CNN in part 2.

    ± load_CNN_part2.py: This Python file is used to load and test the CNN model trained in part2_CNN.py.

  - requirments.txt: Requirements file containing necessary libraries to run the project
    
  - main.py: This Python code is used to run the project overall containing all section.
    
  - "Dataset": This folder is kept empty due to the fact that the MedMNIST api is used to load the BreastMNIST and BloodMNIST datasets.
 
