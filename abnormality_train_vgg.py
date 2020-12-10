from keras.applications.resnet50 import ResNet50 as ResNet, preprocess_input
from keras_preprocessing.image import ImageDataGenerator
import models
import numpy as np
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from sklearn.metrics import classification_report, confusion_matrix
from data import get_data_generators
from utils import *
from keras.models import load_model

batch_size = 8
image_size = 256
epochs = [5, 15]


callbacks = models.get_callbacks()

class_weights = {0: 0.5, 1: 1.0}


train_generator, val_generator, test_generator = get_data_generators(image_size, batch_size, 'GRY', 'abnormality', 'vgg')

# model = models.create_vgg_model(image_size=256, dropout=0.5)


#
# print("Stage 1 - Transfer Learning:")
#
# model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
#
# history = model.fit(train_generator,
#                     steps_per_epoch=train_generator.samples // batch_size,
#                     epochs=epochs[0],
#                     callbacks=callbacks,
#                     validation_data=val_generator,
#                     validation_steps=val_generator.samples // batch_size,
#                     class_weight=class_weights,
#                     verbose=2)
#
# print("First stage done.")

model = load_model("checkpoints/checkpoint-05-0.80.hdf5")

# try:
#     val_loss_history = history.history['val_loss']
#     val_acc_history = history.history['val_accuracy']
#     loss_history = history.history['loss']
#     acc_history = history.history['accuracy']
# except KeyError:
loss_history = []
acc_history = []
val_loss_history = []
val_acc_history = []


# Stage 2:
print("Stage 2 - Fine Tuning:")
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
                    verbose=2)

print("Second stage completed. Have mercy upon my soul!")

try:
    loss_history = np.append(loss_history, history.history['loss'])
    acc_history = np.append(acc_history, history.history['accuracy'])
    val_loss_history = np.append(loss_history, history.history['val_loss'])
    val_acc_history = np.append(acc_history, history.history['val_accuracy'])
except KeyError:
    pass


model.save("model_vgg_abnormality_GRY256.h5")


plot_acc(acc_history, val_acc_history)
plot_loss(loss_history, val_loss_history)

steps = test_generator.n//test_generator.batch_size


y_pred = model.predict_generator(test_generator, steps=steps, verbose=1)
y_test = test_generator.classes
y_pred = np.array([np.argmax(x) for x in y_pred])[0:(steps*batch_size)-1]
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
print(classification_report(y_test, y_pred))
