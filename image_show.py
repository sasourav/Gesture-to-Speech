import cv2
from matplotlib import pyplot

for i in range(1, 11):
    image = cv2.imread('Fresh Image/u/'+str(i-1)+'-1.png')
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(RGB_img, (32, 32))
    pyplot.subplot(3, 4, i)
    pyplot.imshow(img)

# show the plot
pyplot.show()