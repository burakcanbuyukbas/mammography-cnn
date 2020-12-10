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
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

image_size = 128
batch_size = 16
epoch = 30

train_generator, val_generator, test_generator = get_data_generators(image_size=image_size, batch_size=batch_size,
                                                                     data='RGB', task='abnormality', model=None)


callbacks = models.get_callbacks(folder="checkpoints/custom/")

class_weights = {0: 1.0,
                1: 1.0}


model = models.create_custom_model3(size=image_size)


model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])


history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // batch_size,
                    epochs=epoch,
                    callbacks=callbacks,
                    validation_data=val_generator,
                    validation_steps=val_generator.samples // batch_size,
                    class_weight=class_weights,
                    verbose=2)



model.save("model_custom3_abnormality_RGB128.h5")
try:
    plot_loss_from_hist(history)
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()

finally:
    steps = test_generator.n//test_generator.batch_size


    y_pred = model.predict_generator(test_generator, steps=steps, verbose=1)
    y_test = test_generator.classes
    y_pred = np.array([np.round(x) for x in y_pred])[0:((test_generator.samples // batch_size)*batch_size)-1]
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print(classification_report(y_test, y_pred))


