
from matplotlib import pyplot as plt
import skimage
#from skimage import morphology
import cv2

#from skimage import data
from skimage.filter import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import closing, square
from skimage.measure import regionprops
from skimage.color import label2rgb
import skimage.io as ski_io

img = cv2.imread('img_train.tif',0)

cv2.imshow('Image',img)
cv2.waitKey(0)
