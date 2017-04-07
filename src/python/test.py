import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier


# Load an color image in grayscale
#img = cv2.imread('21_training.tif',0)
#img_manual = cv2.imread('21_manual1.tif',0)
#cv2.imshow('Retina', img)

x = np.arange(-10,10,0.1)
y = np.sin(x)*np.exp(np.sin(np.pi*0.5*x))
plt.plot(x,y)


train = x[::10]
train = [[i] for i in train]
label = y[::10]
clf =  MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(1,2), random_state=1)
clf.fit(train,label)

#x_test = x[::5]
#y_test =clf.predict(x_test)

#plt.plot(x_test,y_test,'*r')
#plt.show()

'''
k = cv2.waitKey(0)
if k == 27: # key = ESC
    cv2.destroyAllWindows()
elif k == ord('s'): # key = s
    #cv2.imwrite('retina2.tif',img)
    cv2.destroyAllWindows()
'''
