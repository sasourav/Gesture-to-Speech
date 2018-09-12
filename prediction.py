from __future__ import print_function
import numpy as np
import cv2
import os
import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

batch_size = 128
num_classes = 10
epochs = 50
prediction = 0
pred = 0
data_augmentation = True

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
model.load_weights('weight.h5', by_name=True)
import cv2
arr=[]
prediction=0
img = cv2.imread('E:/ML/s/Fresh Image/t/4 (1).jpg',0)
img1 = cv2.resize(img, (32, 32))
img2 = img1.reshape(1, 32, 32, 1)
print(model.predict(img2))
arr = model.predict(img2)
for i in range(0, 10):
    if arr[0][i] == 1:
        prediction = i
        break
print(prediction)
pred=str(prediction)
from googletrans import Translator
import os
translator = Translator()
translations = translator.translate([pred], dest='bn')
for translation in translations:
    print(translation.origin, ' -> ', translation.text)

from gtts import gTTS
text = pred
targetLanguage = 'bn'
tts = gTTS(text, targetLanguage)
tts.save("9.mp3")
os.system('9.mp3')
# The matrix part
'''from sklearn import metrics
import numpy as np

ytest = [np.where(r == 1)[0][0] for r in y_test]
Y_pred = model.predict(x_test)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(ytest)
print(type(x_test))
print(type(y_test))

print(metrics.confusion_matrix(ytest, y_pred))
print(metrics.f1_score(ytest, y_pred, average='macro'))
print(metrics.mean_absolute_error(ytest, y_pred))
print(metrics.mean_squared_error(ytest, y_pred))'''

# The ROC curve part
'''import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

# Plot linewidth.
lw = 2

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
print(y_test.shape)
print(Y_pred.shape)
for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], Y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), Y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(10)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(10):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= 10

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(["red", "orange", "yellow", "green", "blue", "purple", "black", 'aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(10), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(["red", "orange", "yellow", "green", "blue", "purple", "black", 'aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(10), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Zoom view of Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()'''

# Accuracy and Loss curve
'''from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
                                 # randomly shift images horizontally (fraction of total width)
                                 width_shift_range=0.1,
                                 # randomly shift images vertically (fraction of total height)
                                 height_shift_range=0.1,
                                 shear_range=0.2,  # set range for random shear
                                 zoom_range=0.2,  # set range for random zoom
                                 horizontal_flip=True)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
history_callback = model.fit_generator(datagen.flow(x_train, y_train,
                                                        batch_size=batch_size),
                                           epochs=100,
                                           validation_data=(x_test, y_test),
                                           workers=4)


def plot_history(logger):
    df = pd.DataFrame(logger.history)
    df[['acc', 'val_acc']].plot()
    plt.ylabel("accuracy")
    df[['loss', 'val_loss']].plot(linestyle='--', ax=plt.twinx())
    plt.ylabel("loss")


plot_history(history_callback)
plt.show()'''
