import numpy as np
import tensorflow as tf

import preprocessing
from preprocessing import *
import keras.backend as K
import matplotlib.pyplot as plt
from keras.utils import plot_model
x_train, y_train = preprocessing.DeleteandAugment(x_train, y_train)
print(x_train.shape)
print(y_train.shape)
inputs = tf.keras.layers.Input((len(x_train[0]), len(x_train[0][0])))           #(None,2000,12)

def Jaccard_(y_test, y_pred):           #jaccard 공식활용
    y_test = K.flatten(y_test)          #1차원 변경
    y_pred = K.flatten(y_pred)

    intersection = K.sum(y_test*y_pred)
    total = K.sum(y_test) + K.sum(y_pred)
    union = total-intersection

    Jaccard_score = (intersection+K.epsilon()) / (union+K.epsilon())
    return Jaccard_score

def dice_lossfunction(y_test, y_pred):      #dice공식 활용
    y_test = K.flatten(y_test)              #1차원 변경
    y_pred = K.flatten(y_pred)

    intersection = K.sum(y_test*y_pred)
    union = K.sum(y_test)+K.sum(y_pred)
    dice = (2 * intersection+K.epsilon()) / union + K.epsilon()                 #K.epsilon() = 분모가 0이 되는 것을 방지

    return 1-dice

def UNet(inputs):
    # kernel size = 9, padding = 4(padding=same과 같은 의미)
    c1 = tf.keras.layers.Conv1D(4, kernel_size=9, padding='same')(inputs)       #(None,2000,4)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.ReLU()(c1)
    c1 = tf.keras.layers.Conv1D(4, kernel_size=9, padding='same')(c1)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.ReLU()(c1)
    p1 = tf.keras.layers.MaxPooling1D(pool_size= 2)(c1)                         #(None,1000,4)

    c2 = tf.keras.layers.Conv1D(8, kernel_size=9, padding='same')(p1)           #(None,1000,8)
    c2 = tf.keras.layers.BatchNormalization()(c2)
    c2 = tf.keras.layers.ReLU()(c2)
    c2 = tf.keras.layers.Conv1D(8, kernel_size=9, padding='same')(c2)
    c2 = tf.keras.layers.BatchNormalization()(c2)
    c2 = tf.keras.layers.ReLU()(c2)
    p2 = tf.keras.layers.MaxPooling1D(pool_size=2)(c2)                          #(,500,8)

    c3 = tf.keras.layers.Conv1D(16, kernel_size=9, padding='same')(p2)          #(,500,16)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.ReLU()(c3)
    c3 = tf.keras.layers.Conv1D(16, kernel_size=9, padding='same')(c3)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.ReLU()(c3)
    p3 = tf.keras.layers.MaxPooling1D(pool_size=2)(c3)                          #(,250,16)

    c4 = tf.keras.layers.Conv1D(32, kernel_size=9, padding='same')(p3)          #(,250,32)
    c4 = tf.keras.layers.BatchNormalization()(c4)
    c4 = tf.keras.layers.ReLU()(c4)
    c4 = tf.keras.layers.Conv1D(32, kernel_size=9, padding='same')(c4)
    c4 = tf.keras.layers.BatchNormalization()(c4)
    c4 = tf.keras.layers.ReLU()(c4)
    p4 = tf.keras.layers.MaxPooling1D(pool_size=2)(c4)                          #(,125,32)

    c5 = tf.keras.layers.Conv1D(64, kernel_size=9, padding='same')(p4)          #(,125,64)
    c5 = tf.keras.layers.BatchNormalization()(c5)
    c5= tf.keras.layers.ReLU()(c5)
    c5 = tf.keras.layers.Conv1D(64, kernel_size=9, padding='same')(c5)
    c5 = tf.keras.layers.BatchNormalization()(c5)
    c5= tf.keras.layers.ReLU()(c5)

    #expansive path, padding 3

    u6 = tf.keras.layers.Conv1DTranspose(96, kernel_size=8, strides=2, padding='same')(c5)  #(,250,96)
    u6 = tf.keras.layers.concatenate([u6, c4])                                              #(,250,128(=96+32))
    c6 = tf.keras.layers.Conv1D(32, kernel_size=9, padding='same')(u6)                      #(,250,32)
    c6 = tf.keras.layers.BatchNormalization()(c6)
    c6 = tf.keras.layers.ReLU()(c6)
    c6 = tf.keras.layers.Conv1D(32, kernel_size=9, padding='same')(c6)
    c6 = tf.keras.layers.BatchNormalization()(c6)
    c6 = tf.keras.layers.ReLU()(c6)

    u7 = tf.keras.layers.Conv1DTranspose(48, kernel_size=8, padding='same', strides=2)(c6)  #(,500,48)
    u7 = tf.keras.layers.concatenate([u7, c3])                                              #(, 500,64(=16+48)
    c7 = tf.keras.layers.Conv1D(16,kernel_size=9, padding='same')(u7)                       #(,500,16)
    c7 = tf.keras.layers.BatchNormalization()(c7)
    c7 = tf.keras.layers.ReLU()(c7)
    c7 = tf.keras.layers.Conv1D(16,kernel_size=9, padding='same')(c7)                       #(,500,16)
    c7 = tf.keras.layers.BatchNormalization()(c7)
    c7 = tf.keras.layers.ReLU()(c7)

    u8 = tf.keras.layers.Conv1DTranspose(24,kernel_size=8, padding='same', strides=2)(c7)   #(,1000,24)
    u8 = tf.keras.layers.concatenate([u8, c2])                                              #(,1000,32(=24+8))
    c8 = tf.keras.layers.Conv1D(8,kernel_size=9, padding='same')(u8)                        #(,1000,8)
    c8 = tf.keras.layers.BatchNormalization()(c8)
    c8 = tf.keras.layers.ReLU()(c8)
    c8 = tf.keras.layers.Conv1D(8,kernel_size=9, padding='same')(c8)
    c8 = tf.keras.layers.BatchNormalization()(c8)
    c8 = tf.keras.layers.ReLU()(c8)

    u9 = tf.keras.layers.Conv1DTranspose(12,kernel_size=8, padding='same', strides=2)(c8)   #(,2000,12)
    u9 = tf.keras.layers.concatenate([u9, c1])                                              #(,2000,16)
    c9 = tf.keras.layers.Conv1D(4,kernel_size=9, padding='same')(u9)                        #(,2000,4)
    c9 = tf.keras.layers.BatchNormalization()(c9)
    c9 = tf.keras.layers.ReLU()(c9)
    c9 = tf.keras.layers.Conv1D(4,kernel_size=9, padding='same')(c9)
    c9 = tf.keras.layers.BatchNormalization()(c9)
    c9 = tf.keras.layers.ReLU()(c9)

    outputs = tf.keras.layers.Conv1D(4, kernel_size=1, activation='softmax')(c9)            #(,2000,4)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    y_pred = model.predict(x_train)
    model.compile(optimizer='adam', loss=dice_lossfunction, metrics=[Jaccard_])
    model.summary()

    plt.plot(y_pred[0])
    plt.show()

    return model



def check(model):
    checkpoint_filepath = 'unet_checkpoint.h5'
    checkpointer = tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, verbose=1, save_best_only=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=1, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')
    ]

    history = model.fit(x_train, y_train, validation_split=0.1, batch_size=16, epochs=100, callbacks=callbacks)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    Jaccard_ = history.history['Jaccard_']
    val_Jaccard_ = history.history['val_Jaccard_']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, Jaccard_, 'y', label='Jaccard')
    plt.plot(epochs, val_Jaccard_, 'r', label='val_Jaccard')
    plt.xlabel('Epochs')
    plt.ylabel('Jaccard_')
    plt.legend()
    plt.show()


    plt.plot(x_train[0])
    plt.show()

    plt.plot(y_train[0])
    plt.show()


    test=[x_train, y_train]
    print(test[0])
    for i, ecg in enumerate(test[0]):  # valid_x

        pred = model.predict(np.reshape(ecg,(1,2000,12)))
        true = np.argmax(test[1][i], axis=-1)
        pred = np.argmax(pred[0], axis=-1)

        fig, ax = plt.subplots(3, 1, figsize=(20, 12))

        ax[0].plot(ecg[:, 1])
        ax[0].set_title(f"Input", fontsize=10)

        ax[1].plot(true)
        ax[1].set_title(f"Groud Truth", fontsize=10)

        ax[2].plot(pred)
        ax[2].set_title(f"Prediction", fontsize=10)

        plt.show()


        if i == 1:
            break
    return history