import cv2
import numpy as np

img = cv2.imread(r'./src/Raw-Data-BAT/HFD/II_2_1_HFD.tiff')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

resize = cv2.resize(gray, (700, 500))

cv2.imshow('tes', resize)

cv2.waitKey(0)
