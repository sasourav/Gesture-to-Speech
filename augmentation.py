from __future__ import print_function
from matplotlib import pyplot
import numpy as np
import cv2
import os
import keras

# Initializing
batch_size = 128
num_classes = 10
epochs = 50
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
for i in range(0, 1):
    files = os.listdir('Fresh Image/s/')  # +str(i)+'/') # Reads each images of folders one by one
    for file in files:
        filename = 'Fresh Image/s/' + file
        img = cv2.imread(filename)
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(RGB_img, (32, 32))
        x_train.append(img)
        y_train.append(i)
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

from keras_preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=30,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True)

# fit augmented data
datagen.fit(x_train, augment=True)

for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size=12):
    # Show 10 images
    for i in range(0, 10):
        pyplot.subplot(3, 4, i + 1)
        pyplot.imshow(X_batch[i].reshape(img_rows, img_cols, 3).astype(np.uint8))

    # show the plot
    pyplot.show()
    break
