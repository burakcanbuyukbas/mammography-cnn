import numpy as np
import pandas as pd
import os
from keras_preprocessing.image import ImageDataGenerator
from PIL import Image
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from data import get_data_generators
from sklearn.metrics import classification_report, confusion_matrix
from utils import *

image_size = 128
batch_size = 16


_, _, test_generator = get_data_generators(image_size=image_size, batch_size=batch_size,
                                                                     data='RGB', task='abnormality', model=None)

# load model
# model = load_model('model_custom2_abnormality_RGB128.h5')

#model = load_model('checkpoints/custom/custom1_checkpoint-20-0.85.hdf5')

model = load_model('model_resnet_abnormality_RGB256.h5')

steps = test_generator.n//test_generator.batch_size


y_pred = model.predict_generator(test_generator, steps=steps, verbose=1)
y_test = test_generator.classes[0:len(y_pred)]
y_pred = np.array([np.round(x) for x in y_pred])
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
print(classification_report(y_test, y_pred))


