#!/usr/bin/env python3
from __future__ import absolute_import, print_function

import sys
import random
import argparse

parser = argparse.ArgumentParser(description='Split a dataset in k-folds')
parser.add_argument('dataset',  help='Dataset to split',             nargs='*', default='/dev/stdin')
parser.add_argument('-k',       help='How many folds',               default=None, required=False, type=int)
parser.add_argument('-p',       help='Probability for training set', default=None, required=False, type=float)
parser.add_argument('-s',       help='Random seed',                  default=None, dest='seed', type=int )
parser.add_argument('--output', help='Output path',                  default='splits.csv')
parser.add_argument('--stream',   help='Streaming mode: folds to print. eg. --stream 1,3,4,5', default='', required=None)
parser.add_argument('--stream_e', help='Streaming mode: folds to exclude. eg. --stream_e 2',   default='', required=None)
args = parser.parse_args()

if not args.k and not args.p:
    print("Either -k or -p should be specified", file=sys.stderr)
    exit(1)

if args.seed:
    random.seed( args.seed )

# split strategy

n_folds = args.k
fold    = lambda: random.randint(0, n_folds-1)

if args.p:
    n_folds = 2
    fold    = lambda: int(random.random() < args.p)


# output mode

folds_to_print = map(int,[ x for x in args.stream.split(',') if x ]) 
output_files   = { i: sys.stdout for i in folds_to_print }

if args.stream_e:
    folds_to_ignore = map(int,[ x for x in args.stream_e.split(',') if x]) 
    output_files    = { i: sys.stdout for i in range(n_folds) if i not in folds_to_ignore }

if not output_files:
    print("Writing splits to {}".format( args.output), file=sys.stderr)
    folds_to_print = range(n_folds)
    output_files   = { i: open("{}.{}".format(args.output, i), 'w') for i in folds_to_print }


# splitting

input  = open(args.dataset)

header = input.readline()
for f in output_files.values():
    f.write(header)

for line in input:
    f = fold()
    if f in output_files:
        output_files[f].write(line)

print("Done!", file=sys.stderr)

