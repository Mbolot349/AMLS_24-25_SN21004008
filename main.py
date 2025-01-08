import os
import subprocess
print("=========================part1 executing==========================")
# Construct the path to svm_part1.py
script_path = os.path.join('A', 'svm_part1.py')

# Call the script with python

print("running the SVM Model for part 1")
subprocess.run(['python', script_path])
print("============================================SVM model finished==============================================")



script_path_1 = os.path.join('A', 'random_forest_part1_load.py')
print("running random forest model for task 1, loaded to save time when executing the code")
print("================================See A/random_forest_final.py for model training==================================")
subprocess.run(['python', script_path_1])
print("==================================Random Forest model finished=====================================================================")

script_path_2 = os.path.join('A', 'load.py')
print("Final Deep Learning Model Loading")
print("loading CNN model and testing")
subprocess.run(['python', script_path_2])
print("TASK 1 CNN Results loaded")
print("=================================TASK 1 IS FINISHED ============================================")