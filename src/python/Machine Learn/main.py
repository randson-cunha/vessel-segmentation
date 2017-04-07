
from time import time
from sklearn.preprocessing import StandardScaler
import cv2
from feature_tool import *
from machine_learning import *

def filter_img(clf, img_test,size_kernel):
    #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               size_kernel = 7
    features = get_features_test(img_test, size_kernel)

    scaler = StandardScaler()
    scaler.fit(features)
    X_test = scaler.transform(features)

    img_filtered = clf.predict(X_test)
    return np.reshape(img_filtered,np.shape(img_test))


# Load an color image in grayscale
img_train = cv2.imread('img_train.tif',0)

img_target = cv2.imread('21_manual1.tif',0)

img_test = cv2.imread('img_test.tif',0)

#========= getting features and lables ==================

#remove pixels com pigmento menor que 10
#a imagem se torna binaria
img_train[img_train < 10] = 0
size_kernel = 11

print "Extracting traing features "
features_train, target_pixel = get_features_and_targets(img_train, img_target,size_kernel)


print "features", np.shape(features_train)
print "targets", np.shape(target_pixel)

k = 100
features_train = features_train[::k]
target_pixel = target_pixel[::k]

#========= Testing neural net ==================
clf = get_RNA_classifier(features_train,target_pixel)
#clf = get_svm_classifier(features_train, target_pixel)
#clf = get_nbayes_classifier(features_train, target_pixel)
#clf = get_kmeans_classifier(features_train, target_pixel)
#clf = get_MLPRgressor(features_train,target_pixel)

print "Predicting new inputs..."
t = time()
new_img = filter_img(clf, img_test,size_kernel)
print time() - t, "s elapsed"

cv2.imshow('Img test', img_test)
cv2.imshow('Segmentation', new_img)

cv2.waitKey(0)
cv2.imwrite('Result.tif',new_img)

'''
k =cv2.waitKey(0)
if k == 27: # key = ESC
    cv2.destroyAllWindows()
elif k == ord('s'): # key = s
    #cv2.imwrite('retina2.tif',img)
    cv2.destroyAllWindows()
'''
