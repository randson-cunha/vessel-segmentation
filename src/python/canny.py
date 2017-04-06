import cv2
import numpy as np
#from matplotlib import pyplot as plt

def  nothing():
    pass

def detec_boards(img, A):
    k_hlines = A*np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]])
    k_vlines = A*np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]])
    k_p45 = np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]])
    k_m45 = np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]])
    img2 = cv2.filter2D(img,0,k_hlines) + cv2.filter2D(img,0,k_vlines) + cv2.filter2D(img,0,k_p45) + cv2.filter2D(img,0,k_m45)

    return img2


img = cv2.imread('21_training.tif',0)
cv2.namedWindow('Canny')

cv2.createTrackbar('T1','Canny',1,150,nothing)
cv2.createTrackbar('T2','Canny',1,150,nothing)
cv2.createTrackbar('k','Canny',1,100,nothing)

# aplica filtro gaussiano pra reduzir o ruido
#img = cv2.medianBlur(img,5)

#k_passa_alta = 3*np.array([[-2,-2,-2],[-2,16,-2],[-2,-2,-2]])
#k_sobel_x = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
#k_sobel_y = np.array([[-1,0,-1],[-2,0,2],[-1,0,1]])
#k_sobel = k_sobel_x + k_sobel_y
#img = cv2.filter2D(img,0,k_sobel_x)

while (1):

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    T1 = cv2.getTrackbarPos('T1','Canny')
    T2 = cv2.getTrackbarPos('T2','Canny')
    k = cv2.getTrackbarPos('k','Canny')
    if k == 0:
        k = 1

    #img2 = cv2.medianBlur(img,3)
    #img2 = detec_boards(img, k)
    img2 = cv2.Canny(img,T1,T2)


    #k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k1,k1))
    #k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k2,k2))

    #img2 = cv2.erode(img2,k_erode, iterations= 1)
    #img2 = cv2.dilate(img2,k_dilate,iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
    img2 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)
    #img2= cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("Canny", img2)

cv2.destroyAllWindows()
