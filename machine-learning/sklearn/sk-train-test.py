#!usr/bin/env python
#-*- coding: utf-8 -*-

import sys
import os
print("Python version:", sys.version)

import time
from sklearn import metrics
import numpy as np
import pickle
import matplotlib.pyplot as plt 


#reload(sys)
#sys.setdefaultencoding('utf8')

#linear_regression
def linear_regression(train_x, train_y):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression(normalize=True)
    model.fit(train_x, train_y)
    return model
#BayesianRidge
def bayesian_ridge_regression(train_x, train_y):
    from sklearn.linear_model import BayesianRidge
    model = BayesianRidge()
    model.fit(train_x, train_y)
    return model

# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model

# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10,
                        100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print (para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'],
                gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model

def show_pictures(X, Y):
    rows = 10
    cols = 10
    fig, ax = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True, ) 
    ax = ax.flatten() 
    for nrow in range(rows):
        for ncol in range(cols): 
            img = X[Y ==  ncol][nrow*cols ].reshape(28, 28)
            #img = X[nline*10 + ncol].reshape(28, 28)
            ax[nrow*cols + ncol].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([]) 
    ax[0].set_yticks([]) 
    plt.tight_layout() 
    plt.show()

def show_scatters(actual, predict, title):
    plt.scatter(actual,predict)
    plt.title(title)
    plt.show()

def save_raw_file(X, Y, name):
    np.savetxt(name+'_X.csv',X)
    np.savetxt(name+'_Y.csv', Y)

## only support liner regression
def regression_train_and_test(train_x, train_y, test_x, test_y):
    model_save_file = None
    model_save = {}

    #test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT']
    test_regressions = ['Line','BR']
    regressions = {
                    'BR':bayesian_ridge_regression,
                   'Line': linear_regression
                   }
    
    print ('******************** Data Info *********************')
    print("Training X data shape : ", train_x.shape )
    print("Training Y data shape : ", train_y.shape)

    for regression in test_regressions:
        print ('******************* %s ********************' % regressions[regression].__name__)
        start_time = time.time()
        model = regressions[regression](train_x, train_y)
        training_time = time.time()
        print('training took %fs!' % (training_time - start_time))

        #predict = model.predict(train_x[:10,:])
        predict = model.predict(test_x)
        np.set_printoptions(formatter={'float':'{:0.1f}'.format})
        #print("[Try] Target:\t", test_y[:10])
        #print("[Try] Predict:\t", predict[:10])
        print("[Try] Target:\t", test_y)
        print("[Try] Predict:\t", predict)
        show_scatters(test_y,predict,regressions[regression].__name__)
        
        predict_time = time.time()
        print('predicting took %fs!' % (predict_time - training_time))

        if model_save_file != None:
            model_save[regression] = model

    if model_save_file != None:
        pickle.dump(model_save, open(model_save_file, 'wb'))

def classifier_train_and_test(train_x, train_y, test_x, test_y):
    thresh = 0.5
    model_save_file = None
    model_save = {}

    #test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT']
    test_classifiers = ['NB']
    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'SVMCV': svm_cross_validation,
                   'GBDT': gradient_boosting_classifier                   
                   }
    
    is_binary_class = (len(np.unique(train_y)) == 2)
    print ('******************** Data Info *********************')    
    print("Training X data shape : ", train_x.shape )
    print("Training Y data shape : ", train_y.shape)
    print("Testing X data shape : ", test_x.shape)
    print("Testing Y data shape : ", test_y.shape)

    for classifier in test_classifiers:
        print ('******************* %s ********************' % classifiers[classifier].__name__)
        start_time = time.time()
        model = classifiers[classifier](train_x, train_y)
        training_time = time.time()
        print('training took %fs!' % (training_time - start_time))

        predict = model.predict(test_x)
        predict_time = time.time()
        print('predicting took %fs!' % (predict_time - training_time))

        if model_save_file != None:
            model_save[classifier] = model
        if is_binary_class:
            precision = metrics.precision_score(test_y, predict)
            recall = metrics.recall_score(test_y, predict)
            print ('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
        accuracy = metrics.accuracy_score(test_y, predict)
        print ('accuracy: %.2f%%' % (100 * accuracy))
        measurescore_time = time.time()
        print('measure score took %fs!' % (measurescore_time - predict_time))
        np.set_printoptions(formatter={'float':'{:0.1f}'.format})
        print("[Try] Target:\t", test_y[:10])
        print("[Try] Predict:\t", predict[:10])
        

    if model_save_file != None:
        pickle.dump(model_save, open(model_save_file, 'wb'))

def process_minist_classifier():
    print ('\n\n#### Data: mnist ####')
    #https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz
    #https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz
    #http://deeplearning.net/data/mnist/mnist.pkl.gz
    data_file = "/var/tmp/mnist.pkl.gz"
    #data_file = "C:\\temp\\mnist.pkl.gz"
    
    import gzip
    f = gzip.open(data_file, "rb")
    #train, val, test = pickle.load(f)
    # encoding issue: http://www.mlblog.net/2016/09/reading-mnist-in-python3.html
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train, val, test = u.load()
    f.close()
    train_x = train[0]
    train_y = train[1]
    test_x = test[0]
    test_y = test[1]
    #show_pic(train_x, train_y)
    #save_raw_file(train_x, train_y,"train")
    classifier_train_and_test(train_x, train_y, test_x, test_y )

def process_20newsgroup_classifier():
    print ('\n\n#### Data: 20 news group  ####')
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    #https://ndownloader.figshare.com/files/5975967
    rawdata = datasets.fetch_20newsgroups()
    rawdata_x = rawdata.data
    rawdata_y = rawdata.target
    train_x, test_x, train_y, test_y = train_test_split(rawdata_x, rawdata_y, test_size=0.3)
    classifier_train_and_test(train_x, train_y, test_x, test_y )

def process_digits_classifier():
    print ('\n\n#### Data: digits handwriting recognition  ####')
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    rawdata = datasets.load_digits()
    rawdata_x = rawdata.data
    rawdata_y = rawdata.target
    train_x, test_x, train_y, test_y = train_test_split(rawdata_x, rawdata_y, test_size=0.3)
    classifier_train_and_test(train_x, train_y, test_x, test_y )

def process_iris_classifier():
    print ('\n\n#### Classifier Data: iris ####')
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    rawdata = datasets.load_iris()
    rawdata_x = rawdata.data
    rawdata_y = rawdata.target
    train_x, test_x, train_y, test_y = train_test_split(rawdata_x, rawdata_y, test_size=0.3)
    classifier_train_and_test(train_x, train_y, test_x, test_y )

def process_boston_regression():
    print ('\n\n#### Regression Data: boston house price ####')
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    rawdata = datasets.load_boston()    
    raw_x = rawdata.data
    raw_y = rawdata.target
    train_x, test_x, train_y, test_y = train_test_split(raw_x, raw_y, test_size=0.3)
    regression_train_and_test(train_x,  train_y,test_x, test_y )
    
def process_diabetes_regression():
    print ('\n\n#### Regression Data: diabetes ####')
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    rawdata = datasets.load_diabetes()
    raw_x = rawdata.data
    raw_y = rawdata.target
    train_x, test_x, train_y, test_y = train_test_split(raw_x, raw_y, test_size=0.3)
    regression_train_and_test(train_x, train_y,test_x, test_y )

if __name__ == '__main__':
    #process_minist()
    #process_20newsgroup()
    process_digits_classifier()
    process_iris_classifier()
    process_boston_regression()
    process_diabetes_regression()
