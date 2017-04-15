import numpy as np
import cv2


# Load an color image in grayscale
img = cv2.imread('21_training.tif',0)

#img2 = cv2.Laplacian(img,cv2.CV_64F)
img2 = cv2.Sobel(img,cv2.CV_64F,1,1,ksize=5)
#sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

threshold = 200
[l,c] = np.shape(img)
for i in range(l):
    for j in range(c):
        if img2[i][j] > threshold:
            pass#img2[i][j] = 0


cv2.imshow('Retina', img2)

k = cv2.waitKey(0)
if k == 27: # key = ESC
    cv2.destroyAllWindows()
elif k == ord('s'): # key = s
    cv2.imwrite('retina2.tif',img)
    cv2.destroyAllWindows()
