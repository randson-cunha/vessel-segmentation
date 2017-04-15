#!/usr/bin/python

print
print "checking for nltk"

try:
    import nltk
    print "Nnltk ok"
except ImportError:
    print "you should install nltk before continuing"

print "checking for numpy"
try:
    import numpy
    print "Numpy ok"
except ImportError:
    print "you should install numpy before continuing"

print "checking for scipy"
try:
    import scipy
    print "Scipy ok"
except:
    print "you should install scipy before continuing"

print "checking for sklearn"
try:
    import sklearn
    print "Sklearn ok"
except:
    print "you should install sklearn before continuing"

print "checking for Neurolab..."
try:
    import neurolab as nl
    print "Neurolab ok"
except:
    print "You should install Neurolab before continue"

print "checking for skimage..."
try:
    import skimage
    print "Skimage is ok"
except:
    print "you should install skimage"
