#!/usr/bin/env python3
# vim: syntax=python

import lightgbm as lgb

def load_data(file_name):
    return lgb.Dataset( file_name )

def set_eval_sets( run_params, train, test ):
    run_params['valid_sets']  = (train, test)
    run_params['valid_names'] = ('train', 'eval')

def get_params( ):
    return dict(
        verbosity=-1,
        objective='regression',
        metric='rmse',

        max_depth=-1,
        num_leaves=2**7,
        min_child_samples=1,
        min_child_weight=10,

        subsample=.7,
        colsample_bytree=1.,
        learning_rate=0.1,
        # lambda_l2=0.0,
        # lambda_l1=0.0,
        # max_bin=255,
        # min_data_in_bin=3,
        # zero_as_missing=False,
        # bin_construct_sample_cnt=200000, # set to very large number when its too sparse
    )