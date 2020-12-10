from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Conv2DTranspose, Activation,\
    Concatenate, SeparableConv2D, Input, BatchNormalization, UpSampling2D, GlobalAveragePooling2D
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.models import load_model, Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.applications.resnet50 import ResNet50 as ResNet, preprocess_input as ResNetPreprocess
from keras.applications.mobilenet_v2 import MobileNetV2 as MobileNet, preprocess_input as MobileNetPreprocess
from keras.applications.vgg16 import VGG16 as VGG16, preprocess_input as VGGPreprocess
from keras import layers

def create_ResNet50_model(dropout=0.5, size=256, classes=1):
    base_model = ResNet(input_shape=(size, size, 3), include_top=False, weights='imagenet', pooling='avg')
    x = base_model.output

    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout)(x)

    out = Dense(classes, activation='sigmoid', name='output_layer')(x)
    model = Model(inputs=base_model.input, outputs=out)

    for layer in base_model.layers:
        layer.trainable = False

    return model

def create_vgg_model(image_size=256, dropout=0.5, classes=1):
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


def create_unet_model(image_size=256, classes=2):
    inputs = Input(shape=(image_size, image_size) + (3,))

    # downsampling

    # Entry block
    x = Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    previous_block_activation = x

    for filters in [64, 128, 256]:
        x = Activation("relu")(x)
        x = SeparableConv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = SeparableConv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])
        previous_block_activation = x

    # upsampling

    for filters in [256, 128, 64, 32]:
        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = UpSampling2D(2)(x)

        residual = UpSampling2D(2)(previous_block_activation)
        residual = Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # per-pixel classification layer
    # x = Conv2D(classes, 3, activation="softmax", padding="same")(x)
    x = layers.Flatten()(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(1, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model


def create_custom_model(size=256, dropout_rate=0.5, classes=1):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(size, size, 3)))
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(classes, activation='sigmoid'))
    return model

def get_callbacks(folder="checkpoints/"):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10,
                                   verbose=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=3, verbose=1)

    filepath = folder + "checkpoint-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpointer = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')

    return [reduce_lr, early_stopping, checkpointer]

def create_custom_model1(size=256, dropout_rate=0.5, classes=1):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(size, size, 3)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(classes, activation='sigmoid'))
    return model


def create_custom_model2(size=256, dropout_rate=0.5, classes=1):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(size, size, 3)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(classes, activation='sigmoid'))
    return model


def create_custom_model3(size=256, dropout_rate=0.5, classes=1):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(size, size, 3)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(classes, activation='sigmoid'))
    return model

def create_mobilenet(size=256, dropout_rate=0.5, classes=1):
    base_model = MobileNet(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(512, activation='relu')(x)
    preds = Dense(classes, activation='sigmoid')(x)

    return Model(inputs=base_model.input, outputs=preds)

