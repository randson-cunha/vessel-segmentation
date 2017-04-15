import numpy as np
import cv2
from matplotlib import pyplot as plt

def nothing():
    pass

def open_img(img,k):
    kernel = np.ones((k,k),np.uint8) #cv2.MORPH_OPEN
    kernel = np.array([[0,0,0],[1,1,1],[0,0,0]])
    kernel = np.uint8(np.array([[0,1,0],[0,1,0],[0,1,0]]))
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
    #return img - open_img(img,k)


def toggle_mapping(img,s1,s2):
    mask1 = close_img(img,s1) - img <= img - open_img(img,s2) # close
    mask2 = close_img(img,s1) - img > img - open_img(img,s2)  #open

    img[mask1] = close_img(img,s1)[mask1]
    img[mask2] = open_img(img,s2)[mask2]
    return img

def detec_boards(img, A):
    k_hlines = A*np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]])
    k_vlines = A*np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]])
    k_p45 = np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]])
    k_m45 = np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]])
    img2 = cv2.filter2D(img,0,k_hlines) + cv2.filter2D(img,0,k_vlines) + cv2.filter2D(img,0,k_p45) + cv2.filter2D(img,0,k_m45)

def create_all_trackbar():
    cv2.namedWindow('Control')
    cv2.createTrackbar('k_gauss','Control',0,50,nothing)
    cv2.createTrackbar('k_open','Control',0,50,nothing)
    cv2.createTrackbar('s1','Control',0,255,nothing)
    cv2.createTrackbar('s2','Control',0,255,nothing)
    cv2.createTrackbar('TH','Control',0,50,nothing)
    cv2.createTrackbar('f','Control',0,255,nothing)

# Load an color image in grayscale
img = cv2.imread('21_training.tif')
cv2.namedWindow('Image')
img_gray = cv2.imread('21_training.tif',0)

create_all_trackbar()
print np.shape(img_gray)
print np.shape(img)

img_b, img_g, img_r = cv2.split(img)



while (1):

    k = cv2.getTrackbarPos('k_gauss','Control')
    k = 2*k+1

    k_open = cv2.getTrackbarPos('k_open','Control')
    k_open = k_open*2+1

    s1 = cv2.getTrackbarPos('s1','Control')
    s1 = s1*2 +1

    s2 = cv2.getTrackbarPos('s2','Control')
    s2 = s2*2 +1

    TH = cv2.getTrackbarPos('TH','Control')
    TH = TH*2 + 1

    f = cv2.getTrackbarPos('f','Control')

    #img_b, img_g, img_r = cv2.split(img)
    #get the gree chanel
    img2 = img[:,:,1]
    #img2 = 255 - img2

    img2 = cv2.GaussianBlur(img2,(k,k),0)
    #img2 = toggle_mapping(img2,s1,s2)
    img2 = tophat_image(img2,TH)

    img2 = cv2.GaussianBlur(img2,(s1,s1),0)

    img2[img2 > f] = 255

    cv2.imshow('Image', img2)
    cv2.imshow('Hist',img2.ravel())

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        cv2.imwrite('img_train.tif',img2)
        cv2.destroyAllWindows()
        break
