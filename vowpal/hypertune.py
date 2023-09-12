#!/usr/bin/env python3
from __future__ import print_function

import subprocess as sb
import numpy as np
import scipy.optimize as sc
import math
import argparse
import re
import os
import sys
import tempfile
import multiprocessing as mp

from collections import defaultdict
from multiprocessing.dummy import Pool

parser = argparse.ArgumentParser(description='', usage="%(prog)s 'command' [options]")
parser.add_argument('cmd',     help="Command to interpolate. Tunable options should be marked with '?' and their ranges separated by '|'.  Eg.: vw --binary --loss_function? logistic|hinge --l2? 1e-10|1e-3 --bfgs?")
parser.add_argument('-n',      help='Number of independent runs',         dest='runs',   default=1, type=int)
parser.add_argument('-m',      help='Search method. Default: random',     dest='method', default='random', choices=['random', 'nelder-mead', 'explore'])
parser.add_argument('-b',      help='Number of bins for continuous args', dest='bins',   default=20)
parser.add_argument('-t',      help='Number of threads',                 dest='threads',        default=mp.cpu_count()-2, type=int)
parser.add_argument('--log',   help='Enable log scale for continuous variables which upper/lower ratio exceed this value. Default: 100', dest='log_scale_k', default=100)
parser.add_argument('--max',   help='Maximize instead of minimizing',    dest='maximize',  action='store_true')
parser.add_argument('--debug', help='Break if there is an error and print stack trace', dest='debug',  action='store_true')
# nelder
parser.add_argument('-i',      help='Max number of iterations per run',  dest='max_iterations', default=20, type=int)
parser.add_argument('-k',      help='Minimum improvement for stopping',  dest='threshold',      default=1e-3, type=float)

args       = parser.parse_args()
grammar    = '([^ ]+)\?( *(?!-)[^ ]+)?'


# helpers

def _try(func, x, *args):
    try:
        return func(x,*args)
    except:
        return x

def _to_num(x):
    if '.' in x or 'E' in x or 'e' in x:
        return float(x)
    return int(x)

def _round(x, digits):
    if type(x) != float:
        return x
    val = ("{:.%sf}" % digits).format(x).rstrip('0')
    return val+'0' if val[-1] == '.' else val

def _print(x_vec, prefix='', file=sys.stdout, digits=10):
    pretty_x = [ _try(_round, v, digits)      for v in _inverse(x_vec)  ]
    cmd      = _cmd("{} "*len(x_vec), pretty_x )
    print(prefix+' | ' if prefix else '', cmd, file=file )

def _inverse(x_vec):
    return [ x_inverse[i](v) for i,v in enumerate(x_vec) ]

def _cmd(cmd_tmpl, x_vec):
    args = [ x_tmpl[i](v).format(params[i][0], v) for i,v in enumerate(x_vec) ]
    return cmd_tmpl.format( *args )


# vowpal

def create_cache(cmd_tmpl, cmd_cache=None):
    cache_file = '/tmp/vw_cache.' + str(random.randint(0,100000))
    cache_arg  = ' --cache_file ' + cache_file 

    if '-c' in cmd_tmpl or '--cache_file' in cmd_tmpl:
        cache_arg  = ''
        cache_file = '.cache'
        matches    = re.findall('--cache_file ([^ ]+)', cmd_tmpl)
        cache_file = matches[0] if len(matches) > 0 else '.cache'

    if os.path.exists(cache_file):
        print("- using cache:", cache_file, file=sys.stderr)

    else:
        print("- generating cache: ", cache_file, file=sys.stderr)
        cmd_cache = cmd_cache or cmd_tmpl
        sb.call('{} {} -k --quiet'.format( cmd_cache, cache_arg ), shell=True)

    return cmd_tmpl.replace('-k', '') + cache_arg

def average_loss_cmd(cmd):
    return cmd + " 2>&1 | grep -Po 'average loss = \K([^ ]+)'"


# optimization methods

def objective(x):
    try:
        cmd     = _cmd(cmd_tmpl, _inverse(x))
        output  = sb.check_output(cmd, shell=True, universal_newlines=True).strip()
        score   = float(output) * multiplier
        _print(x, output, file=sys.stderr)
        return score

    except Exception as e:
        _print(x, 'ERROR!', file=sys.stderr)
        if args.debug: print(e, file=sys.stderr)
        return worst_loss

def minimize(x0):
    if args.method == 'nelder-mead':
        res = sc.minimize(objective, x0=x0, method='Nelder-Mead', \
            options={ 'xatol': args.threshold, 'fatol': args.threshold, 'maxiter': args.max_iterations })
        return res.x, res.fun

    # random or hand-picked
    return x0, objective(x0)



# command line and initial points

params   = re.findall(grammar, args.cmd)
cmd_tmpl = re.sub(grammar, '{}', args.cmd)

try:
    x0 = []
    x_link    = [ lambda x: x ] * len(params)
    x_inverse = [ lambda x: x ] * len(params)
    x_tmpl    = [ lambda x: "{} {}" ] * len(params)
    print("\nVariables:", file=sys.stderr)
    
    for i,(p,bounds) in enumerate(params):
        values    = sorted([ _try(_to_num,b) for b in bounds.strip().split('|') ])
        gen       = lambda: np.random.uniform(values[0], values[1], size=args.runs)

        if len(values) == 1:
            desc      = "binary"
            gen       = lambda: np.random.choice(2, size=args.runs)
            tmpl      = lambda b: ( lambda x: "{} %s" % (b) if x == 1 else '' )
            x_tmpl[i] = tmpl(bounds.strip())
            
        elif len(values) > 2:
            desc = "categorical { %s }" %  bounds.strip()
            gen  = lambda: [ values[i] for i in np.random.choice(len(values), size=args.runs) ]

        elif type(values[0]) == int:
            gen  = lambda: np.random.randint(values[0], values[1]+1, size=args.runs)
            desc = "discrete [{},{}]".format(*values)

        elif values[0] and values[-1] / values[0] >= args.log_scale_k:
            x_link[i]    = lambda x: math.log(x,10)
            x_inverse[i] = lambda x: math.pow(10,x)
            values       = tuple(map(x_link[i], values))
            desc         = "continuous [{},{}] (log space)".format( *[ round(i,2) for i in values ] )

        else:
            desc      = "continuous [{},{}]".format(*values)

        if args.method == 'explore' and 'continuous'  in desc:
            gen  = lambda: np.linspace(values[0], values[1], args.bins)

        print(" {} | {}".format(p.ljust(15), desc), file=sys.stderr)
        x0.append( gen() )

    x0 = zip(*x0)

except ValueError:
    print("ERROR! Couldnt apply log(). Is there any 0 in your interval bounds?", file=sys.stderr)
    exit(1)


# optimization

if 'vw' in cmd_tmpl:
    print("\nvw identified", file=sys.stderr)
    cmd_fixed = re.sub(grammar,  '', args.cmd)
    cmd_tmpl  = create_cache(cmd_tmpl, cmd_fixed)
    cmd_tmpl  = average_loss_cmd(cmd_tmpl)

single_run = args.method in ('random', 'explore')
multiplier = -1 if args.maximize else 1

worst_loss = multiplier * 10000000 
best_loss = worst_loss
best_x    = None

pool    = Pool(args.threads)
results = pool.imap_unordered(minimize, x0)
print("\nStarting to tune:", file=sys.stderr)

for i,(x,fx) in enumerate(results, 1):
    if not single_run:
        _print(x, prefix="> "+str(fx), file=sys.stderr)
    if fx < best_loss:
        best_x    = x
        best_loss = fx

if best_x:
    print("\nBest:", best_loss, file=sys.stderr)
    _print(best_x, digits=50)

