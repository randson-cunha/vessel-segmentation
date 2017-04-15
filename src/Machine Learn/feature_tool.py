import cv2
import numpy as np
import sys
import os
from sklearn.preprocessing import StandardScaler
from preprocessing import preprocess_image

path_DRIVE = 'DRIVE/'

use_green_channel = 0
preprocess = 1
reescale = 1

def get_green_channel_from_img(img):
    img_b, img_g, img_r = cv2.split(img)
    return img_g

# if neigborhood_complet is 1, the function leaves outincomplete neigborhoods
def get_neigborhood(img,coord_p,size_kernel, neigborhood_complet = 0):
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
    print "getting new features..."
    for w in range(width):
        for h in range(height):
            neigborhood = get_neigborhood(img_input,(w,h),size_kernel)
            if len(neigborhood) == 0:
                continue
            features.append(neigborhood)
            targets.append(img_target[w][h])

    return features, targets

def get_features_from_test_image(img_test, size_kernel):
    features = []
    width, height = np.shape(img_test)
    for w in range(width):
        for h in range(height):
            neigborhood = get_neigborhood(img_test,(w,h),size_kernel, neigborhood_complet = 0)
            features.append(neigborhood)

    if reescale:
        scaler = StandardScaler()
        scaler.fit(features)
        features = scaler.transform(features)

    return features

#(img_name, img)
def get_train_and_target_images(path_dir):

    path_dir = os.path.join(path_dir,'training')

    train_images = []
    target_images = []
    data_traing_path = os.path.join(path_dir,'images')
    data_target_path = os.path.join(path_dir,'1st_manual_')

    data_train = os.listdir(data_traing_path)
    data_train.sort()

    data_target = os.listdir(data_target_path)
    data_target.sort()

    for train, target in zip(data_train, data_target):
        train_img_path = os.path.join(data_traing_path,train)
        if use_green_channel:
            train_img = cv2.imread(train_img_path)
            train_img = get_green_channel_from_img(train_img)
        else:
            train_img = cv2.imread(train_img_path,0)


        train_img_name = train_img_path.split('/')[-1]

        if preprocess:
            train_img = preprocess_image(train_img)
            img_path = os.path.join('result/', train_img_name.split('.')[0]+'_preprocessed.tif')
            cv2.imwrite(img_path,train_img)

        train_images.append(train_img)

        target_img_path = os.path.join(data_target_path,target)
        target_img = cv2.imread(target_img_path,0)
        target_img_name = target_img_path.split('/')[-1]
        target_images.append(target_img)

    if not len(train_images) == len(target_images):
        print "It may possible some data has been lost"
        return []

    return train_images, target_images

#this method get the green channel of each image
def get_test_and_expected_images(path_dir = 'DRIVE/'):
    path_dir = os.path.join(path_dir,'test')
    features_test_data = []

    test_data_path = os.path.join(path_dir,'images')
    test_data = os.listdir(test_data_path)
    test_data.sort()

    expected_data_path = os.path.join(path_dir,'1st_manual_')
    expected_data = os.listdir(expected_data_path)
    expected_data.sort()

    data_path = zip(test_data,expected_data)
    for test, expected in data_path:

        path_test_img = os.path.join(test_data_path,test)
        if use_green_channel:
            img = cv2.imread(path_test_img)
            img = get_green_channel_from_img(img)
        else:
            img = cv2.imread(path_test_img,0)

        test_image_name = path_test_img.split('/')[-1]
        print "test img", test_image_name
        path_expected_img = os.path.join(expected_data_path,expected)
        expected_image = cv2.imread(path_expected_img,0)
        expected_image_name = path_expected_img.split('/')[-1]
        print "expected img", expected_image_name

        if preprocess:
            img = preprocess_image(img)
            path_image = os.path.join('result/', test_image_name.split('.')[0]+'_preprocessed.tif')
            cv2.imwrite(path_image,img)

        data = {'test_image_name':test_image_name, 'test_image': img, 'expected_image_name':expected_image_name,
                'expected_image':expected_image}
        features_test_data.append(data)

    return features_test_data

def get_whole_features_and_targets(size_kernel):
    print "getting train images and target images..."
    images_train, images_target = get_train_and_target_images(path_DRIVE)
    features = []
    targets = []
    k = 10
    print "getting train and target features..."
    for item in zip(images_train, images_target):
        train, target = item
        features_, targets_ = get_features_and_targets(train, target,size_kernel)
        del train
        del target

        size_features = len(features_)
        print np.shape(features)
        print np.shape(features_)

        features = features + features_[0:size_features:k]
        targets = targets + targets_[0:size_features:k]
        del features_
        del targets_


    if reescale:
        scaler = StandardScaler()
        scaler.fit(features)
        features = scaler.transform(features)

    return features, targets
