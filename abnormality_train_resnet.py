from keras.applications.resnet50 import ResNet50 as ResNet, preprocess_input
from keras_preprocessing.image import ImageDataGenerator
import models
import numpy as np
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from sklearn.metrics import classification_report, confusion_matrix
from data import get_data_generators
from utils import *
from keras.models import load_model

batch_size = 16
image_size = 128
epochs = [2, 10]


callbacks = models.get_callbacks(folder='checkpoints/resnet/')

class_weights = {0: 1.0, 1: 1.0}

train_generator, val_generator, test_generator = get_data_generators(image_size, batch_size, data='RGB',
                                                                     task='abnormality', model='resnet')

model = models.create_ResNet50_model(dropout=0.5, size=image_size)

# model = load_model("checkpoints/checkpoint-01-0.46.hdf5")

print("Stage 1: Transfer Learning...")

model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])

history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // batch_size,
                    epochs=epochs[0],
                    callbacks=callbacks,
                    validation_data=val_generator,
                    validation_steps=val_generator.samples // batch_size,
                    class_weight=class_weights,
                    verbose=1)

print("First stage done.")

try:
    val_loss_history = history.history['val_loss']
    val_acc_history = history.history['val_accuracy']
    loss_history = history.history['loss']
    acc_history = history.history['accuracy']
except KeyError:
    loss_history = []
    acc_history = []
    val_loss_history = []
    val_acc_history = []

# Stage 2:
print("Stage 2: Fine Tuning...")
for layer in model.layers:
    layer.trainable = True

model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])


history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // batch_size,
                    epochs=epochs[1],
                    callbacks=callbacks,
                    validation_data=val_generator,
                    validation_steps=val_generator.samples // batch_size,
                    class_weight=class_weights,
                    verbose=1)

print("Second stage done. Please be good.")

try:
    loss_history = np.append(loss_history, history.history['loss'])
    acc_history = np.append(acc_history, history.history['accuracy'])
    val_loss_history = np.append(val_loss_history, history.history['val_loss'])
    val_acc_history = np.append(val_acc_history, history.history['val_accuracy'])
except KeyError:
    loss_history = []
    acc_history = []
    val_loss_history = []
    val_acc_history = []


model.save("model_resnet_abnormality_RGB128.h5")


plot_acc(acc_history, val_acc_history)
plot_loss(loss_history, val_loss_history)

steps = test_generator.n//test_generator.batch_size


y_pred = model.predict_generator(test_generator, steps=steps, verbose=1)
y_test = test_generator.classes[0:len(y_pred)]
y_pred = np.array([np.round(x) for x in y_pred])
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
print(classification_report(y_test, y_pred))
