import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical, normalize
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.utils.class_weight import compute_class_weight
from medmnist import BloodMNIST, INFO


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


print('\nShapes of images:')
print('Training: ', X_train.shape)
print('Validation: ', X_val.shape)
print('Testing: ', X_test.shape)




# Reshape and normalize the data
X_train, y_train = X_train / 255.0, (y_train)
X_val, y_val = X_val / 255.0, (y_val)
X_test, y_test = X_test / 255.0, (y_test)

# Data augmentation


label_map = {"0": "basophil",
            "1": "eosinophil",
            "2": "erythroblast",
            "3": "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)",
            "4": "lymphocyte",
            "5": "monocyte",
            "6": "neutrophil",
            "7": "platelet"}



#def visualize(X_train, y_train):
#    fig = plt.figure(figsize=(12, 12))
#    for i in range(9):
 #       plt.subplot(330 + 1 + i)
#        plt.imshow(X_train[i], cmap='gray')  # Access the image directly from the NumPy array
 #       plt.title(f"Class {y_train[i][0]}")  # Access the label
  #  plt.show()


from sklearn.utils import class_weight
import numpy as np

# Compute class weights using y_train (labels for training data)
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
print("Xtest",X_test.shape)


X_train=normalize(X_train, axis=1)
X_val=normalize(X_val, axis=1)
X_test=normalize(X_test, axis=1)
#============================ Categorical the labels ===========
y_train=to_categorical(y_train, num_classes=8)
y_test=to_categorical(y_test, num_classes=8)
y_val=to_categorical(y_val,num_classes = 8)

model= Sequential()


model.add(Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 3), activation="relu"))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512)
model.add(activation="relu")
model.add(Dropout(0.5))
model.add(Dense(8))
model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
#=================================== Print the model Summary =============
model.summary()

datagen = ImageDataGenerator(
        rotation_range=70,
        zoom_range = 0,
        width_shift_range=0.2,  
        height_shift_range=0,  
        horizontal_flip=True,  
        vertical_flip=True)
#====================== Fitting the data Augmntation =================

datagen.fit(X_train)

#============================= Fitting the model with flow the data Augmented 
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32,shuffle=True),
    steps_per_epoch = math.ceil(len(X_train) / 32),
    epochs=100,
    validation_data=(X_val, y_val),
    validation_steps=len(X_val) // 32,
    shuffle=True,
   # callbacks=[custom_early_stopping]
)



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

y_T= model.predict(X_train)
y_PT1=np.argmax(y_T, axis=1)
y_TN=np.argmax(y_train, axis=1)
print("Accuracy_train:",accuracy_score(y_PT1, y_TN))
from sklearn.metrics import roc_auc_score

# Calculate the ROC-AUC score
roc_auc = roc_auc_score(y_test, y_pred)
print("ROC-AUC Score: %f" % roc_auc)


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'],color='blue')
plt.plot(history.history['val_accuracy'],color='red')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

plt.show()
# summarize history for loss
plt.plot(history.history['loss'],color='blue')
plt.plot(history.history['val_loss'],color='red')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()



# Save model weights
model.save_weights('weights_task2.weights.h5')
print('Model weights saved to weights_task2.weights.h5')

# Save the entire model in recommended Keras format
model.save('task2.keras')
print('Entire model saved to task2.keras')
