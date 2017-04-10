from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier, MLPRegressor

def get_RNA_classifier(features, target):
    #========= config neural net ==================

    print "config neural net..."
    verbose = True
    max_iter = 3000
    solver =  'sgd'#'adam' #lbfgs, sgd, lbfgs
    hidden_layer_sizes = (81)
    alpha = 1e-6
    random_state = 1
    activation = 'relu'#'tanh'#'logistic'
    learning_rate = 'constant'
    tol = 1e-6

    #========= Training neural net ==================

    clf =  MLPClassifier(solver=solver, activation = activation, hidden_layer_sizes= hidden_layer_sizes,
                        alpha= alpha, random_state=random_state, verbose =  verbose,
                        max_iter = max_iter, learning_rate = learning_rate, tol = tol)

    scaler = StandardScaler()
    scaler.fit(features)
    X_train = scaler.transform(features)

    clf.fit(X_train, target)

    return clf

def get_MLPRgressor(features, target):
    #========= config neural net ==================

    print "config neural net..."
    verbose = True
    max_iter = 1000
    solver =  'adam'#'adam' #lbfgs, sgd, lbfgs
    hidden_layer_sizes = (10,5)
    alpha = 1e-6
    random_state = 1
    activation = 'relu'#'tanh'#'logistic'
    learning_rate = 'adaptive'
    tol = 1e-6

    #========= Training neural net ==================

    clf =  MLPRegressor(solver=solver, activation = activation, hidden_layer_sizes= hidden_layer_sizes,
                        alpha= alpha, random_state=random_state, verbose =  verbose,
                        max_iter = max_iter, learning_rate = learning_rate, tol = tol)

    scaler = StandardScaler()
    scaler.fit(features)
    X_train = scaler.transform(features)

    clf.fit(X_train, target)

    return clf

def get_svm_classifier(features, target):
    clf = svm.SVC(verbose=True, kernel='rbf')

    scaler = StandardScaler()
    scaler.fit(features)
    X_train = scaler.transform(features)

    clf.fit(X_train, target)

    return clf

def get_nbayes_classifier(features, target):
    clf = GaussianNB()

    scaler = StandardScaler()
    scaler.fit(features)
    X_train = scaler.transform(features)

    clf.fit(X_train, target)
    return clf

def get_kmeans_classifier(features, target):

    clf = KMeans(n_clusters=2, max_iter=500, random_state = None, verbose = True)

    scaler = StandardScaler()
    scaler.fit(features)
    X_train = scaler.transform(features)

    clf.fit(X_train,target)

    return clf

def get_hopfield_classifier(features, target):

    import numpy as np
    import neurolab as nl

    # N E R O
    target =  [[1,0,0,0,1,
               1,1,0,0,1,
               1,0,1,0,1,
               1,0,0,1,1,
               1,0,0,0,1],
              [1,1,1,1,1,
               1,0,0,0,0,
               1,1,1,1,1,
               1,0,0,0,0,
               1,1,1,1,1],
              [1,1,1,1,0,
               1,0,0,0,1,
               1,1,1,1,0,
               1,0,0,1,0,
               1,0,0,0,1],
              [0,1,1,1,0,
               1,0,0,0,1,
               1,0,0,0,1,
               1,0,0,0,1,
               0,1,1,1,0]]

    chars = ['N', 'E', 'R', 'O']
    target = np.asfarray(target)
    # Create and train network
    net = nl.net.newhop(target)

    output = net.sim(target)
    print("Test on train samples:")
    for i in range(len(target)):
        print(chars[i], (output[i] == target[i]).all())

    print("\nTest on defaced N:")
    test =np.asfarray([0,0,0,0,0,
                       1,1,0,0,1,
                       1,1,0,0,1,
                       1,0,1,1,1,
                       0,0,0,1,1])
    test[test==0] = -1
    out = net.sim([test])
