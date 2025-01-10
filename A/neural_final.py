import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, jaccard_score, accuracy_score, roc_auc_score, ConfusionMatrixDisplay
from medmnist import BreastMNIST
from medmnist import INFO
from pathlib import Path

info = INFO['breastmnist']

# Load train, validation, and test splits from BreastMNIST
train_data = BreastMNIST(split='train', download=True)
val_data = BreastMNIST(split='val', download=True)
test_data = BreastMNIST(split='test', download=True)

# Convert datasets to numpy arrays
X_train, y_train = np.array(train_data.imgs), np.array(train_data.labels)
X_val, y_val = np.array(val_data.imgs), np.array(val_data.labels)
X_test, y_test = np.array(test_data.imgs), np.array(test_data.labels)

# Reshape and normalize the data
X_train = X_train[..., np.newaxis] / 255.0
X_val = X_val[..., np.newaxis] / 255.0 
X_test = X_test[..., np.newaxis] / 255.0 


label_map = {0: 'malignant', 1: 'benign/normal'}

# Convert numeric labels to text labels for all datasets
y_train_text = np.array([label_map[label[0]] for label in y_train])
y_val_text = np.array([label_map[label[0]] for label in y_val])
y_test_text = np.array([label_map[label[0]] for label in y_test])

# Visualize some training images
#Reshape the flattened images back to 28x28 for visualization
#fig, axes = plt.subplots(1, 5, figsize=(15, 3))
#for i, ax in enumerate(axes.flat):
#    ax.imshow(X_train[i].squeeze(), cmap='gray')  # Reshape to 28x28
#    ax.set_title(f"Label: {y_train_text[i]}")
#    ax.axis('off')  # Turn off the axis for better visualization

#plt.tight_layout()
#plt.show()



y_train_1d = y_train.flatten()

breast_class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_1d),
    y=y_train_1d
)

weights = {0: breast_class_weights[0], 1: breast_class_weights[1]}
print(f"Class weights for imbalance {weights}")





print("X Train: ", X_train.shape)
print("Y Train: ", y_train.shape)

print("X Test: ", X_test.shape)
print("Y Test: ", y_test.shape)

print("Shape of a single image:", X_train[0].shape)



model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(X_train[0].shape),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform',activation='relu',kernel_regularizer='l2'))#))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Conv2D(128, (3,3), kernel_initializer='he_uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))




model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))



model.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.SGD(),
    metrics=['accuracy']
)

model.summary()


custom_early_stopping = EarlyStopping(monitor='val_loss', patience=9 ,min_delta=0.001)#,

history = model.fit(
    X_train,y_train,
    epochs=500,
    class_weight=weights,
    validation_data=(X_val, y_val),
    callbacks=[custom_early_stopping],
    shuffle=True
)


A=model.evaluate(X_test, y_test)



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

y_Tra= model.predict(X_train)

print("Accuracy_train:",accuracy_score(y_train, y_Tra.round()))

from sklearn.metrics import roc_auc_score

# Calculate the ROC-AUC score
roc_auc = roc_auc_score(y_test, y_pred)
print("ROC-AUC Score: %f" % roc_auc)

fig1=plt.plot(history.history['accuracy'],color='blue')
plt.plot(history.history['val_accuracy'],color='red')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

plt.show()
# summarize history for loss
fig2=plt.plot(history.history['loss'],color='blue')
plt.plot(history.history['val_loss'],color='red')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

from pathlib import Path

def save_model(model, model_name):
    # Saving neural network structure
    model_structure = model.to_json()
    # Saving to file
    f = Path(f"{model_name}.json")
    f.write_text(model_structure)
    print(f"Model structure saved to {model_name}.json")

# Save model structure
save_model(model, "Task1_CNN")

# Save model weights
model.save_weights('weights_task1.weights.h5')
print('Model weights saved to weights_task1.weights.h5')

# Save the entire model in recommended Keras format
model.save('task1.keras')
print('Entire model saved to task1.keras')#


