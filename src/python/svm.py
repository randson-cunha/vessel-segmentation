import numpy as np
import cv2
from sklearn import svm

def nothing():
    pass

def open_img(img,k):
    kernel = np.ones((k,k),np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def close_img(img,k):
    kernel = np.ones((k,k),np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def dilate_image(img,k):
    kernel = np.ones((k,k),np.uint8)
    return cv2.dilate(img,kernel,iterations=1)

def erode_image(img,k):
    kernel = np.ones((k,k),np.uint8)
    return cv2.erode(img,kernel,iterations=1)

def tophat_image(img,k):
    kernel = np.ones((k,k),np.uint8)
    return close_img(img,k) - img
    return img - open_img(img,k)

def toggle_mapping(img,s1,s2):
    mask1 = close_img(img,s1) - img <= img - open_img(img,s2) # close
    mask2 = close_img(img,s1) - img > img - open_img(img,s2)  #open

    img[mask1] = close_img(img,s1)[mask1]
    img[mask2] = open_img(img,s2)[mask2]
    return img


def preprocess_img(img):
    img_b, img_g, img_r = cv2.split(img)

    s1 = 3
    s2 = 0
    k_open = 1
    TH = 3

    img2 = img_g

    img2 = open_img(img2,k_open)
    #img2 = toggle_mapping(img2,s1,s2)
    img2 = tophat_image(img2,TH)

    img2[img2 < s1] = 0

    return img2

# Load an color image in grayscale
img = cv2.imread('21_training.tif')
img_test = cv2.imread('22_training.tif')
img_target = cv2.imread('21_manual1.tif',0)

img_p = preprocess_img(img)
img_p_test = preprocess_img(img_test)

clf = svm.SVC(C=1.0, kernel='rbf')

#clf.fit(img_p,img_target)

#img_prediction = clf.predict(img_p_test)

#cv2.imshow('Prediction',img_prediction)

cv2.waitKey(0)
