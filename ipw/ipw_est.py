#!/usr/bin/env run-pypy.sh

import sys
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(description='Treatment effect using diverse methods')
parser.add_argument('-p', help='Propensities file',           default=None, dest='ipw' )
parser.add_argument('-r', help='Regression predictions file', default=None, dest='preds' )
parser.add_argument('-c', help='Propensity clip',             type=float, default=1e-9,  dest='clip' )
parser.add_argument('-e', help='Estimand: att atc ate',       dest='estimand', optional=False )
args = parser.parse_args()

formulas = {
    'dr': {
        'att': lambda y,z,w: ( y*(1-z)*w/(1-w) + m*(z-w)/(1-w) + y*z,              z   ),
        'atc': lambda y,z,w: ( y*z*(1-w)/w     - m*(z-w)/w     - y*(1-z),          1-z ),
        'ate': lambda y,z,w: ( y*z/w - m*(z-w)/w - y*(1-z)/(1-w) - m*(z-w)/(1-w) , 1 ), 
    },

    'ht': {
        'att': lambda y,z,w: ( z*y,           z     ),
        'ate': lambda y,z,w: ( z*y/w,         1     ),
        'atc': lambda y,z,w: ( z*y*(1.0-w)/w, 1-z   ),
    },

    'ipw': {
        'att': lambda y,z,w: ( y*z,           z           ),
        'ate': lambda y,z,w: ( y*z/w,         z/w         ),
        'atc': lambda y,z,w: ( y*z*(1.0-w)/w, z*(1.0-w)/w ),
    }
}


# loading propensities

clip  = lambda z: min(max(z, args.clip), 1.0-args.clip)
ipw   = None
if args.ipw:
    data = open(args.ipw).read().split("\n")
    ipw  = dict([ tuple(l.split(" ")[::-1]) for l in data if l.strip() ])


# aggregation

numerator    = defaultdict(float)
denominator  = defaultdict(float)

for line in sys.stdin:
    z,y,k = [ i.strip("' ") for i in line.strip().split(" ") ]
    y     = max( 0, float(y) )
    w     = clip( float(1.0 if ipw is None else ipw[k]) )

    for _,fs in formulas.items():
        num_inc, den_inc = fs[args.estimand](y, z, w)
        numerator[n]   += num_inc
        denominator[n] += den_inc


# reporting

for k in sorted(numerator.keys()):
    print(k, numerator[k]/denominator[k])

