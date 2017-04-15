
from time import time

import cv2
from feature_tool import *
from machine_learning import *
from skimage.measure import compare_ssim as ssim

def get_accuracy_mse(img1,img2):

    from sklearn.preprocessing import StandardScaler

    if not np.shape(img1) == np.shape(img2):
        print "The images must have the same shape"
        return 0

    #scaler = StandardScaler()
    #img1 = scaler.fit_transform(img1)
    #img2 = scaler.fit_transform(img2)

    width, heigh = np.shape(img1)
    err = np.sum(np.sum((img1 - img2)**2))
    err /= float(width*heigh)

    return err

size_kernel = 7
features_train, targets = get_whole_features_and_targets(size_kernel)

print np.shape(features_train)
print np.shape(targets)

print "traing", np.shape(features_train)
print "targets", np.shape(targets)

k = 10
features_train = features_train[::k]
targets = targets[::k]

print np.shape(features_train)
print np.shape(targets)

#========= Testing neural net ==================
print "traing neural neural network..."
clf = get_RNA_classifier(features_train,targets)


print "Predicting new inputs..."
data_test = get_test_and_expected_images()
for item in data_test:
    print
    print "predicting", item['test_image_name']

    test_image = item['test_image']
    expected_img = item['expected_image']

    input_features = get_features_from_test_image(test_image,size_kernel)
    img_result = clf.predict(input_features)
    img_result = np.reshape(img_result,np.shape(test_image))

    print "Accuracy mse:", get_accuracy_mse(img_result, expected_img)
    print "Accuracy ssim:", ssim(img_result, expected_img)
    print

    del test_image
    del input_features
    new_name = item['expected_image_name'].split('.')[0]
    cv2.imwrite('result/'+ new_name+'_result.tif', img_result)
    del img_result


'''
accuracy = {}
for img in images_test:
    img_name = img[-1]
    img_result = np.zeros((10,10))
    cv2.imwrite('result/' + img_name + "_result",img_result)

    print "Getting accuracy..."
    accuracy = 0#get_accuracy(new_img,img_manual)
'''
