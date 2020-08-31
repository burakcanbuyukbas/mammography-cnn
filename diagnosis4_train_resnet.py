from keras.applications.resnet50 import ResNet50 as ResNet, preprocess_input
from keras_preprocessing.image import ImageDataGenerator
import models
import numpy as np
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from sklearn.metrics import classification_report, confusion_matrix
from data import get_data_generators
from utils import *

batch_size = 16
image_size = 256
epochs = [3, 5, 30]


callbacks = models.get_callbacks()


#opt = Adam()


train_generator, val_generator, test_generator = get_data_generators(image_size, batch_size, 'GRY',
                                                                     'diagnosis-4class', model='resnet')

model = models.create_ResNet50_model(classes=4)



print("Stage 1:")

model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])

history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // batch_size,
                    epochs=epochs[0],
                    callbacks=callbacks,
                    validation_data=val_generator,
                    validation_steps=val_generator.samples // batch_size,
                    verbose=2)

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
# Train on the top layers
print("Stage 2:")
for layer in model.layers[162:]:
    layer.trainable = True


dense_layer = model.layers[-1]
dropout_layer = model.layers[-2]
dense_layer.kernel_regularizer.l2 = 0.01
dropout_layer.rate = .35
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])

history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // batch_size,
                    epochs=epochs[1],
                    callbacks=callbacks,
                    validation_data=val_generator,
                    validation_steps=val_generator.samples // batch_size,
                    verbose=2)

print("Second stage done.")
try:
    loss_history = np.append(loss_history, history.history['loss'])
    acc_history = np.append(acc_history, history.history['accuracy'])
    val_loss_history = np.append(loss_history, history.history['val_loss'])
    val_acc_history = np.append(acc_history, history.history['val_accuracy'])
except KeyError:
    pass

# Stage 3:
print("Stage 3:")
for layer in model.layers:
    layer.trainable = True
dropout_layer.rate = .5
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.00001), metrics=['accuracy'])


history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // batch_size,
                    epochs=epochs[2],
                    callbacks=callbacks,
                    validation_data=val_generator,
                    validation_steps=val_generator.samples // batch_size,
                    verbose=2)

print("Third stage done. Please be good.")

try:
    loss_history = np.append(loss_history, history.history['loss'])
    acc_history = np.append(acc_history, history.history['accuracy'])
    val_loss_history = np.append(loss_history, history.history['val_loss'])
    val_acc_history = np.append(acc_history, history.history['val_accuracy'])
except KeyError:
    pass


model.save("model_resnet_diagnosis4_GRY256.h5")


plot_acc(acc_history, val_acc_history)
plot_loss(loss_history, val_loss_history)

steps = test_generator.n//test_generator.batch_size


y_pred = model.predict_generator(test_generator, steps=steps, verbose=1)
y_test = test_generator.classes
y_pred = np.array([np.argmax(x) for x in y_pred])[0:(steps*batch_size)-1]
print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3]))
print(classification_report(y_test, y_pred))
