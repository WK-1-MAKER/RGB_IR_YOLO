import cv2

from cv_bridge import CvBridge

img = cv2.imread('data/datasets/RGB_images/images/010001.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow('Test Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
