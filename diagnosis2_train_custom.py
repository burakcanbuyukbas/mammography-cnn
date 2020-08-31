from keras.applications.resnet50 import ResNet50 as ResNet, preprocess_input
from keras_preprocessing.image import ImageDataGenerator
import models
import keras
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report, confusion_matrix
from utils import *
from data import get_data_generators

image_size = 256
batch_size = 16
epoch = 30

train_generator, val_generator, test_generator = get_data_generators(image_size=image_size, batch_size=batch_size,
                                                                     data='RGB', task='diagnosis-2class', model='custom')


callbacks = models.get_callbacks()

class_weights = {0: 0.5,
                1: 1.0}

#opt = Adam()

model = models.create_custom_model()
# model.compile(loss='binary_crossentropy',
#               optimizer=opt,
#               metrics=['accuracy'])

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])


history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // batch_size,
                    epochs=epoch,
                    callbacks=callbacks,
                    validation_data=val_generator,
                    validation_steps=val_generator.samples // batch_size,
                    class_weight=class_weights,
                    verbose=2)



model.save("model_custom_diagnosis2_RGB256.h5")


plot_acc_from_hist(history)
plot_loss_from_hist(history)

steps = test_generator.n//test_generator.batch_size


y_pred = model.predict_generator(test_generator, steps=steps, verbose=1)
y_test = test_generator.classes
y_pred = np.array([np.argmax(x) for x in y_pred])[0:((test_generator.samples // batch_size)*batch_size)-1]
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
print(classification_report(y_test, y_pred))