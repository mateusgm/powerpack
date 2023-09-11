#!/usr/bin/env python
from __future__ import print_function

import sys
import math
import numpy as np


# params

sep       = ' '
k         = int(sys.argv[1] or 3)
label_i   = int(sys.argv[2] or 0)
label_ext = len(sys.argv) > 3

# helpers

def parse_row(line, label=None):
    row   = line.strip().split(sep)

    if label is None:
        label = row[label_i]
        row   = row[:label_i] + ( row[label_i+1:] if label_i != -1 else []  )
    
    if ':' in line:
        row = [ v.split(':')[-1] for v in row ]

    return map(float,row), int(label)-1


# init

hits_3      = 0
hits_5      = 0
pos         = 0
next_update = 1

for i, row in enumerate(sys.stdin):
    preds, label = parse_row(row)
    
    ranked_pos = np.argsort( preds )[::-1]
    label_pos  = list(ranked_pos).index( label )

    pos    += label_pos
    hits_3 += label_pos < 3
    hits_5 += label_pos < 5

    if next_update == i:
        print("%d.\t%.3f\t%.3f\t%.3f" % (i, pos*2.0 / i, hits_3*2.0 / i, hits_5*2.0 / i))
        next_update = 2**(math.log(i,2) + 1)
        hits_3      = 0
        hits_5      = 0
        pos         = 0


# labels = pd.read_csv('csvs/targets.csv', names=['i','targets']).set_index('targets')
# label_src = open(sys.argv[3])
# next(label_src) # skip header
# for row, label_row in izip(sys.stdin, label_src):

