#!/usr/bin/env python3
# vim: syntax=python
from __future__ import print_function

import sys
import math
import re
import os
import argparse
import subprocess as sb

from tempfile import NamedTemporaryFile
from itertools import combinations
from collections import defaultdict


parser = argparse.ArgumentParser(description='')
parser.add_argument('model',    help='Vowpal binary model')
parser.add_argument('-m',       help='Mapper script to apply row by row to input', dest='mapper',       default='')
parser.add_argument('-t',       help='Source table in case vw input is on hadoop', dest='table',        default='')
parser.add_argument('-q',       help='Interactions',                               dest='interactions', action='append', default=[])
parser.add_argument('--cache',  help='Use cached .features if it exists',          dest='use_cache',    action='store_true')
args = parser.parse_args()


# params

category_char = '='
space_char    = '__'
feature_file  = ( '' if '/tmp' in args.model else args.model ) + '.features'
readable_file = args.model + '.readable'

print("\nATTENTION:\nAssuming categories are separated by '{}' and spaces are encoded with '{}'\n\n".format(category_char, space_char), file=sys.stderr)


# extracting features

if not args.use_cache or not os.path.exists(feature_file):
    repo_dir = os.environ['MULTITOOLS_HOME']
    if args.table:
        cmd = "spark-easy.sh -f {d}/spark/vw_extract_features.py -a {t} > {f}".format( d=repo_dir, t=args.table, f=feature_file )
    else:
        mapper   = args.mapper+'| ' if args.mapper else ''
        cmd      = "USE_PYPY=1 parallel.sh '{} {}/spark/vw_extract_features.py' | awk '!seen[$0]++' > {}".format( mapper, repo_dir, feature_file ) 
    sb.call(cmd, shell=True)


# mapping

features = defaultdict(list)

for line in open(feature_file, 'r'):
    tokens = line.strip().split('^')
    if len(tokens) == 3:
        tokens = [ '^', tokens[-1] ] 
    if len(tokens) > 1:
        features[ tokens[0] ].append( tokens[1] )

if len(features) == 0:
    print("Error: extract features failed", file=sys.stderr)
    exit()

# generating vw input

vw_input   = []
int_ignore = []
int_lookup = [ (i[0], i[1]) for i in args.interactions ]

def add_vw_line(vw_input, *group):
    vw_input.append( "1.0" )
    for ns, fts in group:
        vw_input.extend( (' |', ns) ) 
        for f in fts:
            w = ':1' if ':' not in f else ''
            vw_input.extend( (' ', f, w)  )
    vw_input.append( "\n" )

for ns1,ns2 in int_lookup:
    fts1 = list(features[ns1])
    fts2 = list(features[ns2])
    for f1 in fts1: add_vw_line(vw_input, (ns1, [f1]), (ns2, fts2))
    int_ignore.extend((ns1,ns2))

for ns,fts in features.items():
    if ns in int_ignore: continue
    add_vw_line(vw_input, (ns, list(fts)))


# generating readable

vw_cmd = "vw --invert_hash {r} -i {m} -t --quiet".format( r=readable_file, m=args.model )
proc = sb.Popen( vw_cmd, shell=True, stdin=sb.PIPE, universal_newlines=True )
proc.communicate(input=''.join(vw_input))
proc.wait()


# parsing to csv

output    = []
values_re = "{}([^:\*{}]+)".format(category_char, category_char)
names_re  = "\^([^:\*{}]+)".format(category_char)
readable  = open(readable_file, 'r')

for i,line in enumerate(readable):
    line = line.strip()

    values = re.findall(values_re, line)
    names  = re.findall(names_re,  line) 
    if 'Constant' in line: names = [ 'Constant' ]
    if len(names) == 0: continue

    coef     = line.split(':')[-1]
    ft_name  = '*'.join(names)
    ft_value = '*'.join(values).replace(',', '__')
    
    if '^' in ft_name:
        ft_name = ft_name.replace('^', '')

    if category_char not in line:
        ft_value = ft_name
        ft_name  = 'numerical'

    output.append( ','.join((ft_name, ft_value, coef)) )

for l in sorted(output):
    print(l.replace(space_char, ' '))

sb.call( 'rm {}'.format(readable_file), shell=True)

# End

