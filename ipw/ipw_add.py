#!/usr/bin/env python3

import sys
import argparse

parser = argparse.ArgumentParser(description='Add IPW weight to vw input')
parser.add_argument('-i',     help='Propensities file',             default=None,  dest='ipw' )
parser.add_argument('-b',     help='Base rate',         type=float, default=1.0,   dest='base_rate' )
parser.add_argument('-c',     help='Propensity clip',   type=float, default=1e-9,  dest='clip' )
parser.add_argument('-l',     help='Weight conditioned on label.', action='store_true', dest='label_cond' )
args = parser.parse_args()

clip = lambda z: min(max(z, args.clip), 1.0-args.clip)

# load predictions

data = open(args.ipw).read().split("\n")
prop = dict([ tuple(line.split(" ")[::-1]) for line in data if line.strip() ])

ipw = {
    '1':  lambda tag: args.base_rate         / clip( float(prop[tag]) ),
    '-1': lambda tag: (1.0 - args.base_rate) / clip( 1.0 - float(prop[tag]) ),
}


# apply it

def get_tag(tokens):
    tag_i  = 1 if "'" in tokens[1] or '|' in tokens[1] else 2
    return tokens[tag_i].split('|')[0].strip("'")

for line in sys.stdin:
    tokens = line.strip().split(' ')
    weight = ipw[ tokens[0] if args.label_cond else '1' ]( get_tag(tokens) )
    print( "%s %.5f '%s/w%.5f %s" % ( tokens[0], weight, tokens[0], weight, line.strip()[line.index('|'):] ) )

