#!/usr/bin/env python3
from __future__ import print_function

import numpy as np
import sys
import re
import os
import argparse

parser = argparse.ArgumentParser(description='Vowpal to libsvm converter')
parser.add_argument('-b', dest='bits',    help='Number of bits for hash size',        type=int, default=28)
parser.add_argument('-m', dest='mapping', help='Prior and final mapping destination', default='.hash')
args = parser.parse_args()

hash_size = 2**args.bits
hash_func = hash
hash_dump = args.mapping

last_id   = 1
lookup    = np.zeros(hash_size, dtype=np.int32)


# load a pretrained lookup

if os.path.exists(hash_dump):
    print("Loading previous hash", file=sys.stderr)
    hashes = np.loadtxt(hash_dump, dtype=(int,int), delimiter=',')
    for h,_id in hashes: lookup[h] = _id
    last_id = hashes[:,1].max() + 1
    print("Starting from", last_id, file=sys.stderr)


# hash the streamed data

def dense_hash(ns, x, w=1):
    global last_id, lookup
    my_hash = hash_func((ns,x)) % hash_size
    my_id   = lookup[my_hash]
    if my_id == 0:
        my_id     = lookup[my_hash] = last_id
        last_id  += 1
    return "%d:%s" % (my_id, str(w))

for line in sys.stdin:
    nss      = line.split('|')
    lbl_wgt  = nss[0].split(' ')
    new_line = [ ':'.join(lbl_wgt[:2]) if len(lbl_wgt) == 3 else lbl_wgt[0] ]
    
    for ns in nss[1:]:
        tokens = ns.split(' ') 
        new_line.extend(dense_hash(tokens[0], *f.split(':')) for f in tokens[1:])
    print(' '.join(new_line))


# save the lookup for further use (e.g. when predicting)

if not os.path.exists(hash_dump):
    with open(hash_dump, 'w') as f:
        non_zeros, = np.where(lookup > 0)
        for i in non_zeros:
            f.write("{},{}\n".format(i, lookup[i]))

