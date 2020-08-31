from matplotlib import pyplot as plt
from keras.metrics import *

def plot_acc(train, val):
    plt.plot(train)
    plt.plot(val)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
def plot_loss(train, val):
    plt.plot(train)
    plt.plot(val)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def plot_acc_from_hist(model):
    plt.plot(model.history['acc'])
    plt.plot(model.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
def plot_loss_from_hist(model):
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def scheduler(epoch, initlr):
    if epoch < 5:
        return initlr
    elif epoch >= 5 > 10:
        return initlr * 0.1
    elif epoch >= 10 > 15:
        return initlr * 0.01
    elif epoch >= 15 > 20:
        return initlr * 0.001
    else:
        return initlr * 0.0001


def get_metrics():
    METRICS = [
        TruePositives(name='tp'),
        FalsePositives(name='fp'),
        TrueNegatives(name='tn'),
        FalseNegatives(name='fn'),
        BinaryAccuracy(name='accuracy'),
        Precision(name='precision'),
        Recall(name='recall'),
        AUC(name='auc'),
    ]

    return METRICS