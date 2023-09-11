#!/usr/bin/env python3

from __future__ import print_function
from collections import defaultdict
import sys
import math
import numpy as np
import inspect
import argparse
import python.performance_helper as hlp

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

def _wrap_metric(name, func):
    args,_,_,_ = inspect.getargspec(func)
    def wrapper(y_true, y_pred, sample_weight=None):
        if 'sample_weight' in args:
            return func(y_true, y_pred, sample_weight=sample_weight)
        return func(y_true, y_pred, sample_weight)
    return wrapper


# metric determination

args      = hlp.cli_arguments().parse_args()

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

