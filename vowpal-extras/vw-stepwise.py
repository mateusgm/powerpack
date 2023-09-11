#!/usr/bin/env python3
from __future__ import print_function

import sys
import os
import re
import math
import pandas as pd
import subprocess as sb
import argparse
import multiprocessing as mp

from itertools import combinations_with_replacement as combinations
from multiprocessing.dummy import Pool as ThreadPool

parser = argparse.ArgumentParser(description='', prefix_chars='-+')
parser.add_argument('+q',         help='Interactions mode',                             dest='interactions')
parser.add_argument('+s',         help='Early stop threshold',                          dest='early_stop', type=float, default=1e-4)
parser.add_argument('+k',         help='How many namespaces to evaluate at each round', dest='k',          type=float, default=0.5)
parser.add_argument('+t',         help='Number of threads',                             dest='threads',    type=int,   default=None)
parser.add_argument('+n',         help='Numer of rounds',                               dest='rounds',     type=int,   default=None)
parser.add_argument('+verbose',   help='Verbose mode',                                  dest='verbose',    action='store_true')
parser.add_argument('+backward',  help='Backward mode',                                 dest='backward',   action='store_true')
args, vw_args = parser.parse_known_args()

# processing

def test_feature_set(args):
    key, fts = args
    fts_str  = [ vw_arg.format(i) for i in fts ]
    cmd      = "{} {} 2>&1 | grep -Po 'average loss = \K([^ ]+)'".format( vw_cmd, " ".join(fts_str) ) )
    output   = sb.check_output(cmd, shell=True, universal_newlines=True )
    return key, float(output)

def test_feature_sets(sets):
    results = pool.map( test_feature_set, sets )
    results = sorted(results, key=lambda x: x[1])
    return results

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


# initialization

vw_cmd = "vw " + ' '.join(vw_args)
vw_arg = "--keep '{}'"
input_s = sys.stdin.readline()
letters = re.findall('\|([^ ])', input_s)

for i in re.findall('--ignore ([^ ]+)', vw_cmd):
    letters.remove(i)

if args.interactions:
    vw_arg = "-q '{}'"
    if args.interactions == ':':
        args.interactions = letters
    letters = [ ''.join(i) for i in combinations(letters, 2) if i[0] in args.interactions or i[1] in args.interactions ]

to_consider = len(letters)
k_consider  = int(args.k*to_consider) if args.k < 1 else int(args.k)

if args.verbose:
    print("Interactions: ",    args.interactions, file=sys.stderr)
    print("{} candidates: ".format(len(letters)), letters, file=sys.stderr)

# eval and iterate

pool     = ThreadPool( args.threads or max(mp.cpu_count()-2, 2) )
features = pd.DataFrame(columns=['feature', 'score'])
vw_cmd   = create_cache(vw_cmd)

for i,_ in enumerate(letters):
    sets     = [ (f, features.feature.tolist() + [f]) for f in letters[:to_consider] ]
    scores   = test_feature_sets(sets)

    if to_consider != k_consider:
        letters = list(list(zip(*scores))[0])
    to_consider = k_consider

    features.loc[i] = scores[0]
    letters.remove( scores[0][0] ) 

    if args.verbose:
        to_report = None if i == 0 else 5
        print("\n".join(map(str,scores[:to_report])) + "\n------------", file=sys.stderr)
    
    if ( i > 0 and abs(features.score[i] - features.score[i-1]) < args.early_stop) or ( args.rounds and i >= args.rounds ):
        break

print(features)
pool.close()
pool.join()

