#!/usr/bin/env python3

from __future__ import print_function
from collections import defaultdict
from os.path import expanduser
import argparse
import sys
import math
import inspect
import argparse
import numpy as np
import scipy.stats as st
import sklearn.metrics        as sk_metrics
import python.metrics_helper as hlp

# helpers

first_print  = []

def _info(msg, force=False, file=sys.stderr):
    global first_print
    if not force and first_print:
        first_print.append(msg)
        return
    print(msg, file=file)
    file.flush()

def _print_headers(metrics):
    global first_print
    for m in first_print:
        _info(m, force=True)
    _info("\n\t|\t\t" + "\t\t\t|\t\t".join([ m for m,_ in metrics ]))
    _info("\t|\t"     + "\t\t|\t"  .join([ "avg\t\tmodel" ] * len(metrics)))
    first_print = False

def _print(i, count, metrics, final=False):
    if type(first_print) == list:
        _print_headers(metrics)

    _format = "{}\t" + ("|\t" +( "{:.4f}  \t" * 2 )) * len(metrics)
    models  = ('avg', 'model')

    avg     = avg_preds if len(avg_preds) != 0 else np.repeat( float(total) / count, len(labels) )
    preds   = dict(avg=avg, model=predictions)
    values  = [ [ m(labels, preds[s], weights) for s in models ] for _,m in metrics ] 
    _info(_format.format( i, *np.array(values).flatten() ))

    if final:
        values = [ "%f" % v[1] for v in values ]
        _info(' '.join(values), file=sys.stdout)


# metrics

def metric_lookup(threshold=0.5):
    # helpers

    decision    = lambda p: (np.array(p) > threshold) + 0
    inverse_log = lambda p: np.exp(p) 
    w_wrapper   = lambda func, transform: ( lambda t,p,w: func(t, transform(p), sample_weight=w) )
    a_wrapper   = lambda func, transform: ( lambda t,p,w: func(t, transform(p), w) )
    prob_clip   = lambda preds: np.clip(preds, 1e-4, 1-1e-4)

    # metrics

    regression = {
        'rmse':    lambda l,p,w: math.sqrt(sk_metrics.mean_squared_error(l,p,w)),
        'mse':     sk_metrics.mean_squared_error,
        'mae':     sk_metrics.mean_absolute_error,
        'r2':      sk_metrics.r2_score,
    }

    probability = {
        'log_loss':       sk_metrics.log_loss,
        'brier':          lambda l,p,w: sk_metrics.brier_score_loss(l, prob_clip(p), w),
        'auc':            sk_metrics.roc_auc_score,
        'neg_binom_ll':   lambda l,p,w: -st.binom( w, p ).logpmf( (w*l).astype('int') ).mean(),
    }


    binary   = {
        'accuracy':  w_wrapper( sk_metrics.accuracy_score, decision ),
        'recall':    w_wrapper( sk_metrics.recall_score, decision ),
        'precision': w_wrapper( sk_metrics.precision_score, decision ),
        'f1':        w_wrapper( sk_metrics.f1_score, decision ),
    }

    log   = dict(
        neg_poisson_ll= lambda l,p,w: -st.poisson(p).logpmf( map(int,l) ).mean(),
        **{ k+'_exp': a_wrapper( f, inverse_log ) for k,f in regression.items() } 
    )

    def _add_type(_type, items):
        return [ (m, (v, _type)) for m,v in items.items() ]

    return dict( \
        _add_type('regression',     regression)  +\
        _add_type('probability',    probability) +\
        _add_type('binary',         binary)      +\
        _add_type('log',            log) )


def _wrap_metric(name, func):
    args,_,_,_ = inspect.getargspec(func)
    def wrapper(y_true, y_pred, sample_weight=None):
        if 'sample_weight' in args:
            return func(y_true, y_pred, sample_weight=sample_weight)
        return func(y_true, y_pred, sample_weight)
    return wrapper

# args

def cli_arguments(prefix='--', positional=True, usage=None ):
    parser  = argparse.ArgumentParser(prefix_chars='-'+prefix, usage=usage)
    metrics = sorted(metric_lookup().keys())
    if positional:
        parser.add_argument( 'metrics', help='Metrics to be computed, separated by spaces',    nargs='+', choices=metrics)
    else:
        for m in metrics: parser.add_argument( prefix[0]+m, action='store_true')
    parser.add_argument( prefix+'threshold',   help='Probability threshold to classify positives',    default=0.5,   type=float)
    parser.add_argument( prefix+'plot_sample', help='Regression only: sample size for scatter plots', default=None,  type=int)
    parser.add_argument( prefix+'plot_tile',   help='Regression only: percentile to filter outliers', default=0.99,  type=float)
    parser.add_argument( prefix+'plot_dir',    help='Plot directory. Default: ~/plots | /mnt/notebooks/plots on gcloud', default=expanduser('~/plots'))
    parser.add_argument( prefix+'logistic',    help='Apply sigmoid on predictions',                   default=False, action='store_true')
    parser.add_argument( prefix+'training',    help='Training mode: compute loss since last print',   default=False, dest='training_mode', action='store_true')
    parser.add_argument( prefix+'first',       help='Starting printing from row k (so all labels can be observed)', default=128, type=int)
    parser.add_argument( prefix+'name',        help='Idenfifier for this model',                      default='',)
    return parser

# metric determination

args      = cli_arguments().parse_args()

to_report = [ ]
lookup    = hlp.metric_lookup( threshold=args.threshold ) 
for m in args.metrics:
    if m in lookup:
        to_report.append( (m, _wrap_metric(m, lookup[m][0])) )

if len(to_report) == 0:
    _info("Error: you should specify at least one metric", force=True)
    exit(1)

if args.training_mode:
    _info("Training mode: metrics calculated since last print")
else:
    _info("Holdout mode:  metrics calculated since beginning")

p_link    = None
t_link    = None
if args.logistic:
    p_link = lambda z: 1.0/(1.0 + math.exp(-z))
    t_link = lambda z: min(max(int(z),0),1)

# running
    
weights, labels, predictions = [], [], []
total, count = .0, .0
next_update  = args.first
avg_preds    = []

for i, line in enumerate( sys.stdin ):
    row = line.strip().split(" ")
    tag = row[1].split('/')
   
    estimate = float(row[0]) + 1e-9
    truth    = float(tag[0])
    weight   = 1.0
    for t in tag[1:]:
        if t[0] == 'w': weight   = float(t[1:])
        if t[0] == 'a': avg_preds.append(float(t[1:]))
   
    if t_link:
        truth = t_link(truth)
    
    if p_link:
        estimate = p_link(estimate)

    predictions.append(estimate)
    labels.append(truth)
    weights.append(weight)

    total += truth*weight
    count += weight 

    if i+1 == next_update:
        _print( i+1, count, to_report )
        next_update = 2**(math.log(next_update,2)+1)
        if args.training_mode:
            total, count = .0, .0
            labels, predictions, weights = [], [], []

if not first_print:
    _print( i+1, count, to_report )
    hlp.visualize_performance( labels, predictions, weights, sample_size=args.plot_sample, tile=args.plot_tile, model=args.name, save_dir=args.plot_dir)
    _info("Performance charts saved to {}".format( args.plot_dir ))
    _print( i+1, count, to_report, final=True )

