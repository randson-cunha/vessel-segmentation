import cv2
import numpy as np
import sys
sys.path.append('../')
from morphology import *


def get_green_channel_from_img(img):
    img_b, img_g, img_r = cv2.split(img)
    return img_g

# if neigborhood_complet is 1, the function leaves outincomplete neigborhoods
def get_neigborhood(img,coord_p,size_kernel, neigborhood_complet = 1):
    i0 = coord_p[0] - size_kernel/2
    j0 = coord_p[1] - size_kernel/2
    i = i0
    j = j0
    width, height = np.shape(img)
    neigborhood = []
    for i in range(i0,i0 + size_kernel):
        for j in range(j0,j0 + size_kernel):
            if i < 0 or j < 0:
                if neigborhood_complet:
                    return []
                else:
                    pixel = 0
            elif i >= width or j >= height:
                if neigborhood_complet:
                    return []
                else:
                    pixel = 0
            else:
                pixel = img[i][j]
            neigborhood.append(pixel)
    return neigborhood



def get_features_and_targets(img_input, img_target,size_kernel):
    features = []
    targets = []
    width, height = np.shape(img_input)
    for w in range(width):
        for h in range(height):
            if img_input[w][h] < 255:
                pass#continue

            neigborhood = get_neigborhood(img_input,(w,h),size_kernel)
            if len(neigborhood) == 0:
                continue
            features.append(neigborhood)
            targets.append(img_target[w][h])

    return features, targets

def get_features_test(img_input, size_kernel):
    features = []
    width, height = np.shape(img_input)
    for w in range(width):
        for h in range(height):
            neigborhood = get_neigborhood(img_input,(w,h),size_kernel, neigborhood_complet = 0)
            features.append(neigborhood)

    return features


def nothing():
    pass

def create_all_trackbar():
    cv2.namedWindow('Control')
    cv2.createTrackbar('k_gauss','Control',0,50,nothing)
    cv2.createTrackbar('k_open','Control',0,50,nothing)
    cv2.createTrackbar('f','Control',0,255,nothing)

def preprocess_img(img, img_out_name):

    create_all_trackbar()

    while (1):

        k = cv2.getTrackbarPos('k_gauss','Control')
        k = 2*k+1

        k_open = cv2.getTrackbarPos('k_open','Control')
        k_open = k_open*2+1

        f = cv2.getTrackbarPos('f','Control')

        img2 = get_green_channel_from_img(img)

        im2 = open_img(img,7)
        TH = 7
        img2 = tophat_image(img2,TH)

        img2 = cv2.GaussianBlur(img2,(k,k),0)

        img2[img2 > f] = 255

        img2 = open_img(img2,k_open)
        cv2.imshow('Preprocessing image',img2)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.imwrite(img_out_name +'.tif',img2)
            cv2.destroyAllWindows()
            break

        return img2
