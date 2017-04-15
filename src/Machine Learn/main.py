
from time import time
from sklearn.preprocessing import StandardScaler
import cv2
from feature_tool import *
from machine_learning import *

def filter_img(clf, features, size_kernel):

    #scaler = StandardScaler()
    #scaler.fit(features)
    #X_test = scaler.transform(features)

    img_filtered = clf.predict(X_test)
    return np.reshape(img_filtered,np.shape(img_test))

def get_accuracy(img1,img2):

    if not np.shape(img1) == np.shape(img2):
        print "The images must have the same shape"
        return 0

    accuracy = pow(1.0*(img1 - img2),2)
    return 1.0/(1 + sum(sum(accuracy)))

size_kernel = 9

features_train, targets = get_whole_features_and_targets(size_kernel)

print np.shape(features_train)
print np.shape(targets)

print "traing", np.shape(features_train)
print "targets", np.shape(targets)

k = 1
features_train = features_train[::k]
targets = targets[::k]

print np.shape(features_train)
print np.shape(targets)

#========= Testing neural net ==================
print "traing neural neural network..."
clf = get_RNA_classifier(features_train,targets)
#clf = get_svm_classifier(features_train, target_pixel)
#clf = get_nbayes_classifier(features_train, target_pixel)
#clf = get_kmeans_classifier(features_train, target_pixel)
#clf = get_MLPRgressor(features_train,target_pixel)

#t = time()
#new_img = filter_img(clf, img_test,size_kernel)
#print time() - t, "s elapsed"

print "Predicting new inputs..."
data_test = get_test_and_expected_images()
for item in data_test:
    print "predictiong", item['test_image_name']
    test_image = item['test_image']
    expected_img = item['expected_image']
    input_features = get_features_from_test_image(test_image,size_kernel)
    img_result = clf.predict(input_features)
    img_result = np.reshape(img_result,np.shape(test_image))
    new_name = item['expected_image_name'].split('.')[0]
    cv2.imwrite('result/'+ new_name+'_result.tif', img_result)


'''
accuracy = {}
for img in images_test:
    img_name = img[-1]
    img_result = np.zeros((10,10))
    cv2.imwrite('result/' + img_name + "_result",img_result)

    print "Getting accuracy..."
    accuracy = 0#get_accuracy(new_img,img_manual)
'''


#cv2.waitKey(0)
#cv2.imwrite('Result.tif',new_img)

'''
k =cv2.waitKey(0)
if k == 27: # key = ESC
    cv2.destroyAllWindows()
elif k == ord('s'): # key = s
    #cv2.imwrite('retina2.tif',img)
    cv2.destroyAllWindows()
'''
