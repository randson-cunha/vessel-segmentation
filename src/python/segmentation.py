import numpy as np
import cv2


def nothing():
    pass

img = cv2.imread('21_training.tif',0)

w,h = np.shape(img)

cv2.namedWindow("Controller")
cv2.createTrackbar('p1x',"Controller",0,w,nothing)
cv2.createTrackbar('p2x',"Controller",0,w,nothing)
cv2.createTrackbar('p1y',"Controller",0,h,nothing)
cv2.createTrackbar('p2y',"Controller",0,h,nothing)

while (1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27: # key = ESC
        cv2.destroyAllWindows()
        break
    elif k == ord('s'): # key = s
        #cv2.imwrite('retina2.tif',img)
        cv2.destroyAllWindows()
        break

    p1x = cv2.getTrackbarPos('p1x','Controller')
    p1y = cv2.getTrackbarPos('p1y','Controller')
    p2x = cv2.getTrackbarPos('p2x','Controller')
    p2y = cv2.getTrackbarPos('p2y','Controller')


    width = abs(p2x - p1x)
    height = abs(p2y - p1y)

    img2 = img
    for i in range(w):
        for j in range(h):
            if (i >= p1x and i <= p2x) and ( j >= p1y and j <= p2y):
                img2[i][j] = 255
            else:
                img2[i][j] = img[i][j]

    cv2.imshow('Retina', img2)
