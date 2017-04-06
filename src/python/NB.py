import sys
from time import time
from sklearn.naive_bayes import GaussianNB
import cv2
import numpy as np

#features_train, features_test, labels_train, labels_test = preprocess()

#img_test = cv2.imread('22_test.tif',0)

img_train = cv2.imread('21_img_train.tif')
img_target = cv2.imread('21_manual1.tif',0)

l,c = np.shape(img_target)

np.reshape(img_target,[l*10,c/10])

#cv2.namedWindow('Image')
#cv2.imshow('Image',img_target)
#cv2.waitKey(0)

#classify = GaussianNB()
#classify.fit(img_train, img_target)
