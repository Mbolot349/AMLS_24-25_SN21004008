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
print("For training please see A/neaural_final.py")
subprocess.run(['python', script_path_2])
print("TASK 1 CNN Results loaded")
print("=================================TASK 1 IS FINISHED ============================================")

print("================================EXECUTING TASK 2============================")
print("===================loading Random Forest model================================ ")
print("for training please see RF_part2.py, here the model is loaded to save time")
script_path_3 = os.path.join('B', 'random_forest_part2_load.py')
subprocess.run(['python', script_path_3])
print("==================RANDOM_FOREST PART 2 HAS FINISHED=======================")

print("=========================================TASK 2 CNN=============================================")
print("For CNN training please see B/part2_CNN.py")
script_path_4 = os.path.join('B', 'load_CNN_part2.py')
subprocess.run(['python', script_path_4])
print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-Project execution Complete=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")