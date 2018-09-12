# Flip images vertically
from __future__ import print_function
from matplotlib import pyplot
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
epochs = 50
prediction = 0
pred = 0

# input image dimensions
img_rows, img_cols = 32, 32

# The following 4 list are organized like this x_train = Training Images, y_train = Corresponding Labels of the train images.
# Same goes for x_test,y_test

x_train = []
y_train=[]
x_test=[]
y_test=[]
for i in range(0,1):
    files = os.listdir('Fresh Image/s/')#+str(i)+'/') # Reads each images of folders one by one
    for file in files:
        filename = 'Fresh Image/s/'+file
        img = cv2.imread(filename)
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(RGB_img,(32,32))
        x_train.append(img)
        y_train.append(i)
'''for i in range(0,10):
    files = os.listdir('2/test_set/'+str(i)+'/')# Reads each images of folders one by one

    for file in files:
        filename = '2/test_set/'+str(i)+'/'+file
        img = cv2.imread(filename)
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(RGB_img,(32,32))
        x_test.append(img)
        y_test.append(i)'''
x_train=np.asarray(x_train)
y_train=np.asarray(y_train)
x_test=np.asarray(x_test)
y_test=np.asarray(y_test)

'''if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255'''
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


from keras_preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
                                 # randomly shift images horizontally (fraction of total width)
                                 width_shift_range=0.1,
                                 # randomly shift images vertically (fraction of total height)
                                 height_shift_range=0.1,
                                 shear_range=0.2,  # set range for random shear
                                 zoom_range=0.2,  # set range for random zoom
                                 horizontal_flip=True)  # randomly flip images


# fit parameters from data
datagen.fit(x_train,augment=True)

for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size=12):
    # Show 9 images
    for i in range(0, 10):
        pyplot.subplot(3,4,i+1)
        pyplot.imshow(X_batch[i].reshape(img_rows, img_cols, 3).astype(np.uint8))

    # show the plot
    pyplot.show()
    break
'''print(x_train[0])
images = range(0,9)
for i in images:
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(x_train[i])
    pyplot.show()'''

