from keras.applications.resnet50 import ResNet50 as ResNet, preprocess_input as resnet_preprocess
from keras.applications.vgg16 import vgg16, preprocess_input as vgg_preprocess
from keras_preprocessing.image import ImageDataGenerator
import numpy as np

def get_data_generators(image_size=256, batch_size=16, data='GRY', task='diagnosis-2class', model='resnet'):
    if(task=='diagnosis-class'):
        classmode = 'categorical'
    else:
        classmode = 'binary'

    if(model=='resnet'):
        preprocess_func = resnet_preprocess
    elif(model=='vgg'):
        preprocess_func = vgg_preprocess
    else:
        preprocess_func = None

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1.0 / 255.0)
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.3,
        rescale=1.0 / 255.0)

    # test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        directory="data/"+task+"/"+data+"/train",
        target_size=(image_size, image_size),
        class_mode=classmode,
        batch_size=batch_size
    )

    val_generator = val_datagen.flow_from_directory(
        directory="data/"+task+"/"+data+"/test",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode=classmode,
        subset='training'
    )

    test_generator = val_datagen.flow_from_directory(
        directory="data/"+task+"/"+data+"/test",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode=classmode,
        subset='validation',
        shuffle=False)

    return train_generator, val_generator, test_generator
