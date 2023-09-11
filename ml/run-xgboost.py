#!/usr/bin/env python3
# vim: syntax=python
from __future__ import print_function

import xgboost  as xgb
import argparse
import numpy as np
import math
import sys
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('train_set', nargs='?', default='')
parser.add_argument('-t',         required=False, dest='test_set')
parser.add_argument('-n',         default=500, dest='n_iterations', type=int)
parser.add_argument('--tune',   required=False, type=int)
parser.add_argument('--seed',    required=False, type=int, default=0)
parser.add_argument('--train',   required=False)
parser.add_argument('--predict', required=False)
parser.add_argument('--leafs',   required=False)
parser.add_argument('--ensemble', required=False, nargs='+')
parser.add_argument('--loss',     required=False, default='squared')
args = parser.parse_args()


# helpers

def info(*msg):
    print( "[%s]" % (datetime.now().strftime('%H:%M:%S')), *msg, file=sys.stderr)
    sys.stderr.flush()


def load_data(file_name):
    return xgb.DMatrix( file_name )

def set_eval_sets( run_params, train, test, early_stopping_rounds=3 ):
    run_params['evals'] = [ (train, 'train'), (test, 'eval') ]
    run_params['early_stopping_rounds'] = 3

def get_objective():
    if args.loss == 'squared':
        return 'reg:linear'
    return 'binary:logistic'

def get_metric():
    if args.loss == 'squared':
        return [ 'rmse' ]
    return [ 'auc', 'logloss' ]

def get_params():
    return dict(
        objective=get_objective(),
        eval_metric=get_metric(),
        tree_method='hist',  # hist, approx
        
        min_child_weight=8,
        max_depth=12,
        min_split_loss=0,

        subsample=1.,
        colsample_bytree=1.,
        learning_rate=0.3,
        
        # reg_lambda=0.0,
        # reg_alpha=0.0,
        
        # max_bin=256,
        # scale_pos_weight=1,
        seed=args.seed,
        silent=True,
    )



# params

gbm        = globals()['xgb']
gbm_params = get_params()
run_params = dict(
    num_boost_round=args.n_iterations,
    verbose_eval=10,
)

# load data

dtrain = None
if args.train_set:
    info( "Reading training set" )
    dtrain = load_data( args.train_set )

dtest = None
if args.test_set:
    info( "Reading test set" )
    dtest      = load_data( args.test_set )
    set_eval_sets( run_params, dtrain, dtest ) 


# take action

if args.train:
    # train and save model
    info( "Starting to train" )
    bst = gbm.train( gbm_params, dtrain, **run_params )
    bst.save_model( args.train )
    info( bst.eval( dtest or dtrain ) )

if args.predict:
    info( "Starting to predict" )
    bst = xgb.Booster( model_file=args.predict )
    preds = bst.predict(dtest)
    for p,t in zip(preds, dtest.get_label()):
        print("%s %s" % (p,t))
    info( bst.eval(dtest) )

if args.ensemble:
    preds = []
    for m in args.ensemble:
        dtmp = load_data( m )
        bst  = xgb.Booster( gbm_params )
        bst.load_model( m.replace('test.0', 'train.0.xgb') )
        preds.append( bst.predict(dtmp) )

    total, count = [.0]*len(preds), [.0]*len(preds)
    for x in zip(dtmp.get_label(), dtmp.get_weight(), *preds):
        truth, weight, ps = x[0], x[1], x[2:]
        print("%s\t%s/w%s" % (t, sum(ps)/len(ps), w))
        # for i in range(len(ps)):
            # pred = sum(ps[:i+1])/(i+1)
            # total[i] += weight*( (truth-pred)**2 )
            # count[i] += weight
    # print([ math.sqrt(t/c) for t,c in zip(total,count) ])

if args.leafs:
    bst = gbm.Booster( gbm_params )
    bst.load_model( args.leafs )
    preds = bst.predict(dtest, pred_leaf=True)
    pass
    # TODO print stuff

if args.tune:
    # for high lr, define optimal boosting iterations (with cv)
    # tune the intra tree params (max depth, ...) and then reg params (alpha, lambda)
    # lower the lr
    import math
    import pandas as pd
    import sys
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    def obj(space):
        my_params = dict(list(gbm_params.items()) + list(space.items()))
        results = { } 
        bst = gbm.train( my_params, dtrain, evals_result=results, **run_params )
        metric  = get_metric()[-1]
        return results['train'][metric][-1], results['eval'][metric][-1]
    
    space ={
        'max_depth':        [ 6, 8, 10, 12, 14 ],
        'min_child_weight': [ 2, 4, 8, 16, 32 ],
        'subsample':        [ 0.5, 0.75, 1.0 ],
        'colsample_bytree': [ 0.3, 0.5, 0.7, 1.0 ],
        # 'learning_rate':    [ 0.01, 0.05, 0.1, 0.3   ],
    }

    results = pd.DataFrame([], columns=['train', 'test']+sorted(space.keys()))
    for i in range( args.tune ):
        my_space     = { k: np.random.choice(r) for k,r in space.items() }
        train, test  = obj(my_space)
        results.loc[i] = [ train, test ] + [ v for k,v in sorted(my_space.items()) ]
        info( "\n", results ) 

    print( "FINAL: \n", results.sort_values('test', ascending=True) )

    for k in space.keys():
        print( results[['train','test',k]].groupby(k).mean() )

