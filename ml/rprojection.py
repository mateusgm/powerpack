#!/usr/bin/env python3
__author__ = 'lbernardi + mmelo'

import gzip
import sys
import fileinput
import argparse
from collections import defaultdict
import math
import random
from math import sqrt
import os.path
#from bitarray import bitarray

parser = argparse.ArgumentParser(description='Compute a random projection of the input data set.')
parser.add_argument('-i', help='Input steram of values. CSV. Default is standard input.', default='/dev/stdin', dest='input_file')
parser.add_argument('-iz', help='Input is gzipped.', dest='iz', action='store_true')
parser.add_argument('-o', help='Output file. Default is standard output.', default='/dev/stdout', dest='output_file')
parser.add_argument('-oz', help='Gzip output', dest='oz', action='store_true')
parser.add_argument('-t', help='Target variable, untouched. Default -1', dest='target_i', default=-1)
parser.add_argument('-s', help='Separator. Default: ,', dest='sep', default='\t')
parser.add_argument('-k', help='Dimensionality of the reduced space', dest='dim', type=int, default=100)
parser.add_argument('-ri', help='Random Index file name',  dest='ri_file_name')
parser.add_argument('-d', help='How sparse the matrix should be? Expressed as the denominator of the density: 1/d', dest='density_factor', default=3)
parser.add_argument('-p', help='Seed for running in parallel mode', dest='parallel_seed', type=int)
parser.add_argument('--vw_i',  help='Input is vw format: logistic|normal',  dest='vw_input')
parser.add_argument('--vw_o',  help='Output is vw format', dest='vw_output', action='store_true')
parser.add_argument('--svm',   help='Output in svm format', dest='svm_output', action='store_true')
parser.add_argument('--num',   help='Print also numerical features', dest='include_num', action='store_true')

args = parser.parse_args()

input_file_name, gzipped_input, output_file_name, gzipped_output, target_index, sep, dim, ri_file_name = args.input_file, args.iz, args.output_file, args.oz, args.target_i, args.sep, args.dim, args.ri_file_name
density_factor = args.density_factor
parallel_seed  = args.parallel_seed
svm_output     = args.svm_output
vw_input, vw_output = args.vw_input, args.vw_output

def randomVector(column_seed, k):
    if args.parallel_seed:
        random.seed((parallel_seed, column_seed))
    #rv = k*bitarray([0])
    #rv = k*[0]
    rv = []
    for i in range(k):
        x = random.randint(1, 2*density_factor)
        if x == 1:
            rv.append(i)
        if x == 2:
            rv.append(-i)

    return rv


if gzipped_input:
    input = gzip.open(input_file_name)
else:
    input = open(input_file_name)

if gzipped_output:
    output = gzip.open(output_file_name, 'w')
else:
    output = open(output_file_name, 'w')


header = input.readline().strip().split(sep)
if target_index <0:
    target_index += len(header)

mask = '%s,'*dim

num_last  = 0
num_index = {}
index = {}

if ri_file_name:
    if os.path.isfile(ri_file_name):
        ri_file = open(ri_file_name)
        for line in ri_file:
            row = [ x.strip() for x in line.rstrip().split(',') ]
            fn, fv, ri = row[0], row[1], map(int, row[2:])
            index[(fn, fv)]=ri
        ri_file.close()
    ri_file = open(ri_file_name, 'a')

q = math.sqrt(density_factor)/math.sqrt(dim)
for n, line in enumerate(input):
    row = [ x.strip() for x in line.strip().split(sep) ]
    projection = [0]*dim

    if vw_input:
        nss     = line.strip().split('|')
        lbl_wgt = line.strip().split(' ')[:3]
        lbl_wgt[0] = lbl_wgt[0] if vw_input != 'logistic' or lbl_wgt[0] == '1' else '0'
        lbl_wgt = lbl_wgt[:2] if ("'" not in lbl_wgt[1] and '|' not in lbl_wgt[1]) else [ lbl_wgt[0] ]
        row, header = [],[]
        num_vals    = [ 0 ] * 1000
        for ns in nss[1:]:
            cat = [ f.split('=') for f in ns.split(' ')[1:] if f.strip() and '=' in f ]
            for n,v in cat:
                header.append(n)
                row.append(v)
            
            if args.include_num:
                num = [ f.split(':') for f in ns.split(' ')[1:] if f.strip() and ':' in f ]
                for n,v in num:
                    if n not in num_index:
                        num_index[ n ] = num_last
                        num_last += 1
                    num_vals[ num_index[n] ] = float(v)
        num_vals = num_vals[:num_last]

    for fi, fv in enumerate(row):
        if not vw_input and fi == target_index:
            lbl_wgt = [ row[target_index] ]
            continue

        try:
            key = (header[fi], fv)
            r = index[key]

        except KeyError:
            r = randomVector(key, dim)
            index[key]=r

            if  ri_file_name:
                ri_file.write('%s,%s,%s\n' % (key[0], key[1], ','.join(map(str, r))))

        for ri in r:
            absri = abs(ri)
            projection[absri] += q*(1 if ri > 0 else -1)

    if svm_output:
        fts = ' '.join([ "%d:%.4f" % x for x in enumerate(projection)      if x[1] != 0])
        num = ''
        if args.include_num:
            num = ' '.join([ "%d:%.4f" % x for x in enumerate(num_vals, dim) if x[1] != 0 ])
        output.write("{} {} {}\n".format(':'.join(lbl_wgt), fts, num))

    elif vw_output:
        fts = ' '.join([ "%d:%.4f" % x for x in enumerate(projection) if x[1] != 0])
        output.write("{} |f {}\n".format(' '.join(lbl_wgt), fts))

    else:
        output.write(mask % tuple(projection))
        output.write(','.join(lbl_wgt)+'\n')

