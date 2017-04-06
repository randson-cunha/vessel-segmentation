import numpy as np
import cv2

def nothing():
    pass

def open_img(img,k):
    kernel = np.ones((k,k),np.uint8) #cv2.MORPH_OPEN
    #kernel = np.uint8(np.array([[0,1,0],[0,1,0],[0,1,0]]))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def close_img(img,k):
    k = 7
    kernel = np.uint8(np.array([[0,1,0],[0,1,0],[0,1,0]]))
    kernel = np.ones((k,k),np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def dilate_image(img,k):
    kernel = np.ones((k,k),np.uint8)
    return cv2.dilate(img,kernel,iterations=1)

def erode_image(img,k):
    kernel = np.ones((k,k),np.uint8)

    return cv2.erode(img,kernel,iterations=1)

def get_vertical_line_filter():
    return  np.uint8(np.array([
                        [0,0,1,1,1,0,0],
                        [0,0,1,1,1,0,0],
                        [0,0,1,1,1,0,0],
                        [0,0,1,1,1,0,0],
                        [0,0,1,1,1,0,0],
                        [0,0,1,1,1,0,0],
                        [0,0,1,1,1,0,0]]))

def get_horizontal_line_filter():
    return np.uint8(np.array([
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [1,1,1,1,1,1,1],
                        [1,1,1,1,1,1,1],
                        [1,1,1,1,1,1,1],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0]]))

def teste(img,k):
    k = 7

    inc = 360/7
    ang = 0
    for i in range(7):
        rows,cols = np.shape(kernel_hl)
        M = cv2.getRotationMatrix2D((cols/2,rows/2),ang,1)
        kernel_hl = cv2.warpAffine(kernel_hl,M,(cols,rows))

        img = img - close_img(img,kernel_hl)
        ang = (ang + inc)%270
        print ang

    return img

def tophat_image(img,k):
    k = 7
    kernel_hl = get_horizontal_line_filter
    ang = [0,30, 60,120, 150]
    img = close_img(img,kernel_hl) - img

    return img

def reconstruct(img1,img2,n):
    kernel_hl = get_horizontal_line_filter( )
    k = 1
    kernel = np.ones((k,k),np.uint8)
    img_ = np.zeros(np.shape(img2))
    for i in range(n):
        img2 = cv2.dilate(img2,kernel_hl,iterations=1) - img1

    return img2

def create_all_trackbar():
    cv2.namedWindow('Control')
    cv2.createTrackbar('k_gauss','Control',0,50,nothing)
    cv2.createTrackbar('k_open','Control',0,50,nothing)
    cv2.createTrackbar('f','Control',0,255,nothing)

# Load an color image in grayscale
img = cv2.imread('21_training.tif')
cv2.namedWindow('Image')

create_all_trackbar()

img_b, img_g, img_r = cv2.split(img)

while (1):

    k = cv2.getTrackbarPos('k_gauss','Control')
    k = 2*k+1

    k_open = cv2.getTrackbarPos('k_open','Control')
    k_open = k_open*2+1

    f = cv2.getTrackbarPos('f','Control')

    #img_b, img_g, img_r = cv2.split(img)
    #get the gree chanel
    img2 = img[:,:,1]

    im2 = open_img(img,7)
    TH = 7
    img2 = tophat_image(img2,TH)


    img2 = cv2.GaussianBlur(img2,(k,k),0)
    #img2 = cv2.GaussianBlur(img2,(k_open,k_open),0,7.0/4)

    img2[img2 > f] = 255

    img2 = open_img(img2,k_open)

    cv2.imshow('Image', img2)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        cv2.imwrite('img_train.tif',img2)
        cv2.destroyAllWindows()
        break
