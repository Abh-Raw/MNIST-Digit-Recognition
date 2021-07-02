import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.utils import np_utils #print_summary
import pandas as pd
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import keras.datasets.mnist
from keras.models import load_model

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

image_x = 28                    #image size in dataset
image_y = 28

y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
train_y = np_utils.to_categorical(y_train)      #Changed labels to categorical
test_y = np_utils.to_categorical(y_test)
X_train = X_train.reshape(X_train.shape[0], image_x, image_y, 1)
X_train = np.asarray(X_train).astype(np.float64)        #converted arrays to float
X_test = X_test.reshape(X_test.shape[0], image_x,image_y, 1 )
X_test = np.asarray(X_test).astype(np.float64)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

def keras_model(image_x,image_y):       #Model building(Model - Convolutional Neural Network)
    num_of_classes = 10
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5,5), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size = (5, 5), strides = (5, 5), padding = 'same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_of_classes, activation='softmax'))      #Used softmax activation for probability of each label
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "mnist.h5"          #filepath for saved model in .h5 format
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_beat_only=True, mode='max')
    callbacks_list = [checkpoint1]

    return model, callbacks_list

model, callbacks_list = keras_model(image_x, image_y)

# Fit the test data as vlidation in trained model
model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=2, batch_size=64, callbacks=callbacks_list)
scores = model.evaluate(X_test, test_y, verbose=0)
print("CNN Error: %.2f%%" % (100 - scores[1] * 100))

#Printed model details
model.summary()
model.save('mnist.h5')