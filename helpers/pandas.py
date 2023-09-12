
import pandas as pd
import numpy as np
from itertools import combinations


def downsample(X, y, proportion=None, claz=0):
    if proportion is None or y.value_counts()[claz] > proportion:
        return X, y
    y            = y.reset_index(drop=True)
    values       = y[y == claz]
    correction   = (len(values) - (proportion*len(y))) / (1 - proportion)
    negative_smp = np.random.choice(values.index, len(values)-np.abs(correction), replace=False)
    selected     = np.append(y[y != claz].index, negative_smp)
    return X[selected], y[selected]


def add_combinations(X, degrees, features=None):
    if features is None:
        features = X.columns
    combs = combinations(list(range(features.size)), degrees)
    for features_i in combs:
        feature_vals = [ hash(tuple(v)) for v in X[features].ix[:,features_i].values ]
        feature_name = "comb_" +  '_'.join([str(x) for x in features_i])
        X.loc[:,feature_name] = feature_vals
    return X


def describe(data):
    columns = [ 'type', 'Nas', '0s', 'min', 'max', 'mean', '.25', '.50', '.75', '.95', 'uniq' ]
    desc    = pd.DataFrame(index=data.columns, columns=columns)
    
    for c in data.columns:
        if(data[c].dtype == 'object'):
            _0s  = (data[c] == '').sum()
            vals = data[c].value_counts()
            min  = "%s: %d" % ( vals.index[-1], vals.values[-1] )
            max  = "%s: %d" % ( vals.index[ 0], vals.values[ 0] )
            uniq = vals.shape[0]
        else:
            _0s  = (data[c] == 0).sum()
            vals = data[c]
            min  = vals.min()
            max  = vals.max()
            uniq = ''
        
        _25, _50, _75, _95 = vals.dropna().quantile([ .25, .50, .75, .95 ]).apply(f)
        mean, std, cnt, na = vals.mean(), vals.std(), data[c].shape[0] * 1.0,  data[c].isnull().sum()
        desc.loc[c]        = ( data[c].dtype, f(na / cnt), f(_0s / cnt), min, max, f(mean), _25, _50, _75, _95, uniq )
    
    return desc.sort_values(by='type')