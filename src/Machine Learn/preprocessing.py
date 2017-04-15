import cv2
import numpy as np
import os

def close_img(img):
    k = 7
    kernel = np.ones((k,k),np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def tophat_image(img):
    img = close_img(img) - img
    return img

def preprocess_image(image):
    k = 1
    threshold = 6

    image = tophat_image(image)
    image = cv2.GaussianBlur(image,(k,k),0)
    image[image > threshold] = 255

    return image

path = 'DRIVE/'

path = os.path.join(path,'training/images')

for item in os.listdir(path):
    image_name = item.split('/')[-1].split('.')[0]+"_preprocessed.tif"
    path_image = os.path.join(path,item)
    img = cv2.imread(path_image)
    preprocessed_image = preprocess_image(img)
    path_preprocessed_image = os.path.join('result/',image_name)
    cv2.imwrite(path_preprocessed_image,preprocessed_image)
