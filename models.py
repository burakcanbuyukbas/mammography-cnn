from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, Activation,\
    Concatenate
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.models import load_model, Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.applications.resnet50 import ResNet50 as ResNet, preprocess_input as ResNetPreprocess
from keras.applications.vgg16 import VGG16 as VGG16, preprocess_input as VGGPreprocess

def create_ResNet50_model(dropout=0.5, size=256, classes=2):
    resnet = ResNet(weights='imagenet', include_top=False,
                   input_shape=(size, size, 3), pooling='avg')
    x = resnet.output
    x = Dropout(dropout)(x)
    preds = Dense(classes, activation='softmax')(x)
    model = Model(inputs=resnet.input, outputs=preds)
    for layer in resnet.layers:
        layer.trainable = False

    return model

def create_vgg_model(image_size=256, dropout=0.5, classes=2):
    vgg = VGG16(weights='imagenet',
                include_top=False,
                input_shape=(image_size, image_size, 3))

    model = Sequential()
    model.add(vgg)
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(classes, activation='sigmoid'))

    # Freeze the convolutional base
    vgg.trainable = False

    return model

def create_custom_model(size=256, drops=(0.35, 0.5), classes=2):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=3, activation='relu', padding='same', input_shape=(size, size, 3)))
    model.add(Conv2D(16, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(16, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(126, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(126, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(126, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(drops[0]))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(drops[1]))
    model.add(Dense(classes, activation='softmax'))
    return model

def get_callbacks():
    early_stopping = EarlyStopping(monitor='val_loss', patience=10,
                                   verbose=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=3, verbose=1)

    filepath = "checkpoints/checkpoint-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpointer = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')

    return [reduce_lr, early_stopping, checkpointer]

