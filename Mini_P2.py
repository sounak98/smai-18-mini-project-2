import os
import sys
import pickle
import numpy as np
import pdb
from collections import defaultdict
import random 
import time

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import *

from functools import wraps
from time import time as _timenow 
from sys import stderr

scaler = StandardScaler()

def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def load_cifar():
    trn_data, trn_labels, tst_data, tst_labels = [], [], [], []
    
    for i in trange(1):
        batchName = './data/data_batch_{0}'.format(i + 1)
        unpickled = unpickle(batchName)
        trn_data.extend(unpickled[b'data'])
        trn_labels.extend(unpickled[b'labels'])
    
    unpickled = unpickle('./data/test_batch')
    tst_data.extend(unpickled[b'data'])
    tst_labels.extend(unpickled[b'labels'])
    return trn_data, trn_labels, tst_data, tst_labels

def image_prep(image):
    scaler.fit(image)
    processed_image = scaler.transform(image)
    return processed_image

def reduce_dim(images, labels, method):
    if method == 'pca':
        pca = PCA(n_components=30)
        imgs = pca.fit_transform(images)
        print("The Dimensions of the images after PCA are", imgs[0].shape)
        return imgs
    
    if method == 'lda':
        lda = LinearDiscriminantAnalysis(n_components=200)
        imgs = lda.fit(images, labels).transform(images)
        print("The Dimensions of the images after LDA are", imgs[0].shape)
        return imgs
    
    if method == 'raw':
        print("The Dimensions of the raw images are", images[0].shape)
        return images

def classify(X, Y, method):    
    if method == 'poly':
        clf = svm.SVC(kernel='poly', gamma='scale')
        clf.fit(X, Y)
        print("Softmargin linear SVM Model is Prepared")

    if method == 'RBF':
        clf = svm.SVC(kernel='rbf', gamma='scale')
        clf.fit(X, Y)
        print("RBF kernel SVM Model is Prepared")

    if method == 'logistic':
        clf = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=1000)
        clf.fit(X, Y)
        print("Logistic Regression Model is Prepared")

    if method == 'MLP':
        clf = MLPClassifier(max_iter=1000)
        clf.fit(X, Y)
        print("MLP model is prepared")

    if method == 'CART':
        clf = DecisionTreeClassifier()
        clf.fit(X, Y)
        print("Decision Tree Trained")

    return clf

def test(model, test_data, test_labels):
    predictions = model.predict(test_data)
    print("Accuracy:",accuracy_score(test_labels, predictions))
    print("F1:", f1_score(test_labels, predictions, average='weighted'))

def main():
    dim_reduce = ["pca", "lda", "raw"]
    models = ["logistic", "RBF", "poly", "CART", "MLP"]

    if len(sys.argv) == 3 and sys.argv[1] in dim_reduce and sys.argv[2] in models:
        X, y, _X, _y = load_cifar()
        N = 10000

        train_imgs = {}
        test_imgs = {}

        train_imgs[sys.argv[1]] = reduce_dim(X[:N], y[:N], method=sys.argv[1])
        test_imgs[sys.argv[1]] = reduce_dim(_X, _y, method=sys.argv[1])

        model = classify(train_imgs[sys.argv[1]], y[:N], method=sys.argv[2])
        test(model, test_imgs[sys.argv[1]], _y)
    
    else:
        print("Arguments are not correct")

if __name__ == '__main__':
    main()