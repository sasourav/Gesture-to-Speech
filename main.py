from __future__ import print_function
import numpy as np
import cv2
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 100
prediction = 0
pred = 0

# input image dimensions
img_rows, img_cols = 32, 32

# The following 4 list are organized like this x_train = Training Images, y_train = Corresponding Labels of the train images.
# Same goes for x_test,y_test

x_train = []
y_train = []
x_test = []
y_test = []
for i in range(0, 10):
    files = os.listdir('2/train_set/' + str(i) + '/')  # Reads each images of folders one by one
    for file in files:
        filename = '2/train_set/' + str(i) + '/' + file
        img = cv2.imread(filename, 0)
        img = cv2.resize(img, (32, 32))
        x_train.append(img)
        y_train.append(i)
for i in range(0, 10):
    files = os.listdir('2/test_set/' + str(i) + '/')  # Reads each images of folders one by one

    for file in files:
        filename = '2/test_set/' + str(i) + '/' + file
        img = cv2.imread(filename, 0)
        img = cv2.resize(img, (32, 32))
        x_test.append(img)
        y_test.append(i)
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
model.save_weights('weight.h5')
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
# model.load_weights('weight.h5',by_name=True)
from sklearn.metrics import confusion_matrix
import numpy as np

ytest = [np.where(r == 1)[0][0] for r in y_test]
Y_pred = model.predict(x_test)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(ytest)

print(confusion_matrix(ytest, y_pred))
