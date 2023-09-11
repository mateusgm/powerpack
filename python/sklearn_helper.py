from __future__ import absolute_import, print_function

import pandas as pd
import numpy  as np

from sklearn.metrics import *



#**************************** feature engineering

def label(X, categorical=[]):
    labels = dict()
    for c in range(len(categorical)):
        labels[c] = preprocessing.LabelEncoder()
        X[categorical[c]] = labels[c].fit_transform(X[categorical[c]])
    return X, labels

def one_hot(X, categorical=[]):
    cat_mask = np.in1d(X.columns, categorical)
    ohc = preprocessing.OneHotEncoder(categorical_features=cat_mask, handle_unknown='ignore' )
    return ohc.fit_transform(X).tocsr()

def one_hot_columns(ohc, X_old, labels, categorical):
    cat_mask = np.in1d(X_old.columns, categorical)
    features = [ "%s=%s" % ( categorical[c], str(labels[c].classes_[i]) )
            for c in range(len(ohc.feature_indices_)-1)
            for i in range(ohc.feature_indices_[c+1] - ohc.feature_indices_[c]) ]
    features = np.append( features, X_old.columns[~cat_mask] )
    return features

def feature_group(x):
    return x.split('=')[0]
    # return re.match('([\w_]+)=?', x).groups()[0]

    
#**************************** deployment 

def export( model, features, string=False, sorted=False ):        
    coefs = model.coef_ if hasattr(model, 'coef_') else model.feature_importances_
    intercept = model.intercept_ if hasattr(model, 'intercept_') else "tree"
    lst = list(zip( features, np.array([coefs]).flatten() ))
    if sorted:
        lst = sorted( lst,  key = lambda x: -np.abs(x[0]) )  
    lst = [("intercept", intercept )] + lst
    if string:
        lst = [ ["%s %.3f"] % (name, c) for name, c in lst ]
        return "%s\n  %s" % ( str(intercept), "\n  ".join(lst) )
    return pd.DataFrame(lst, columns=[ 'features', 'value' ])


