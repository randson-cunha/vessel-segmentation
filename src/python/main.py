import numpy as np
import cv2


# Load an color image in grayscale
img = cv2.imread('21_training.tif')

cv2.imshow('Retina', img)

k = cv2.waitKey(0)
if k == 27: # key = ESC
    cv2.destroyAllWindows()
elif k == ord('s'): # key = s
    cv2.imwrite('retina2.tif',img)
    cv2.destroyAllWindows()
