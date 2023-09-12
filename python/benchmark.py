#!/usr/bin/env python3
from __future__ import print_function

import sys
import os
import timeit
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('commands', help='Commands to benchmark', nargs='+')
parser.add_argument('-n',       help='How many rounds to run', dest='rounds', default=5, type=int)
parser.add_argument('--quiet',  help='Dont let the commands print anything', action='store_true')
args = parser.parse_args()

print("cmd\tmin\tavg\tmax")
for i,c in enumerate(args.commands):
    if args.quiet:
        c = '({}) > /dev/null 2>&1'.format(c)
    to_time  = "os.system('{}')".format(c)
    measures = timeit.repeat(to_time, setup='import os', repeat=args.rounds, number=1)
    print("%d\t%.2f\t%.2f\t%2.2f" % (i, min(measures), sum(measures)/args.rounds, max(measures)))
