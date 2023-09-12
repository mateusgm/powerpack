#!/usr/bin/env run-pypy.sh
import sys
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(description='IPW weighted mean')
parser.add_argument('-i', help='Propensities file',            default=None, dest='ipw' )
parser.add_argument('-n', help='Population  size', type=int,   default=None, dest='n' )
parser.add_argument('-c', help='Propensity clip',  type=float, default=1e-9,  dest='clip' )
args = parser.parse_args()

clip  = lambda z: min(max(z, args.clip), 1.0-args.clip)
ipw   = None
if args.ipw:
    data = open(args.ipw).read().split("\n")
    ipw  = dict([ tuple(l.split(" ")[::-1]) for l in data if l.strip() ])

counts  = defaultdict(float)
totals  = defaultdict(float)
weights = {
    'obs': lambda _: 1.0,
    'ate': lambda w: 1.0/w,
    'atc': lambda w: (1.0-w)/w,
}

for line in sys.stdin:
    v,k = [ i.strip("' ") for i in line.strip().split(" ") ]
    w   = clip( float(1.0 if ipw is None else ipw[k]) )
    v   = max( 0, float(v) )

    for i,f in weights.items():
        totals[i] += v * f(w)
        counts[i] += f(w)

print(totals['obs']/counts['obs'], end=' ')
if ipw:
    # weighted
    print(totals['atc']/counts['atc'], end=' ')

    # n-m
    print(totals['atc']/(args.n - counts['obs']), end=' ')
