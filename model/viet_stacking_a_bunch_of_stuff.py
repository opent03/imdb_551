from model.pre3 import retrieve_and_pre, dump_to_csv
from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd

def get_meta_features(classifiers, X_train, y_train, X_test, y_test):
    meta_features_train = []
    meta_features_test = []
    for clf in classifiers:
        print(str(clf))
        clf.fit(X_train, y_train)
        pred_train = clf.predict(X_train)
        pred_test = clf.predict(X_test)
        meta_features_train.append(pred_train)
        meta_features_test.append(pred_test)
        scr = accuracy_score(y_test, pred_test)
        print("accuracy score: %f" % scr)

    return np.asarray(meta_features_train).T, np.asarray(meta_features_test).T


def predict_meta_features(classifiers, X_test):
    meta_features_test = []
    for clf in classifiers:
        print(str(clf))
        pred_test = clf.predict(X_test)
        meta_features_test.append(pred_test)

    return np.asarray(meta_features_test).T


def main():
    X, y, final, names, lol = retrieve_and_pre(fromsave=True, tfidfpca=True, n_components=20000)
    y = np.asarray(y)
    print(X.shape[0], y.shape[0])
    assert X.shape[0] == y.shape[0]
    X_train, X_test, y_train, y_test = train_test_split(X.todense(), y, test_size=0.2, shuffle=True, random_state=5)

    """ STACKING PROCESS """
    """
    ARCHITECTURE: 
        Column1 : LinearSVM
        Column2 : Logistic Regression
        Column3 : Adaboost
        Column4 : KNN
    """

    # model 1
    sgdclf = LinearSVC(C=0.5, verbose=1, max_iter=2000)
    lregclf = LogisticRegression(solver='saga', penalty='l2')
    adaclf = XGBClassifier(n_estimators=500)
    rfclf = RandomForestClassifier(n_estimators=500, max_depth=3)
    naiveclf = GaussianNB()
    mlpclf = MLPClassifier(hidden_layer_sizes=(400,), activation='relu', verbose=True)

    baseclfs = [sgdclf, lregclf, adaclf, rfclf, mlpclf]

    meta_features_train, meta_features_test = get_meta_features(baseclfs, X_train, y_train, X_test, y_test)
    #np.save('meta_X_train', meta_features_train)
    #np.save('meta_X_test', meta_features_test)
    #meta_features_train = np.load('meta_X_train.npy')
    #meta_features_test = np.load('meta_X_test.npy')
    #print(meta_features_train.shape, meta_features_test.shape)

    xgb = XGBClassifier(n_estimators=1000, max_depth=3,silent=False)
    xgb.fit(meta_features_train, y_train)
    meta_pred = xgb.predict(meta_features_test)

    scr = accuracy_score(y_test, meta_pred)
    f1scr = f1_score(y_test, meta_pred)
    print("Accuracy score: %f, f1-score: %f" % (scr, f1scr))

    meta_features_lol = predict_meta_features(baseclfs, lol)
    meta_pred_lol = xgb.predict(meta_features_lol)
    scr = accuracy_score(y, meta_pred_lol)
    f1scr = f1_score(y, meta_pred_lol)
    print("Accuracy score: %f, f1-score: %f" % (scr, f1scr))
    tn, fp, fn, tp = confusion_matrix(y, meta_pred_lol).ravel()
    print("true neg: %i\tfalse pos: %i\nfalse neg: %i\ttrue pos: %i\n" % (tn, fp, fn, tp))

    print("precision: %f; recall: %f" % ((tp / (tp + fp)), (tp / (tp + fn))))

    meta_features_final = predict_meta_features(baseclfs, final)
    dump_to_csv(xgb, final, names, 'stacking_bs')

if __name__ == '__main__':
    main()
