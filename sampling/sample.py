#!/usr/bin/env python3
# vim: syntax=python
from __future__ import print_function

import sys
import random
import argparse

parser = argparse.ArgumentParser(description='Out of core dataset sampling without replacement')
parser.add_argument('input', help='Input',           nargs='?', default='/dev/stdin')
parser.add_argument('-p',    help='Probability',     required=False, type=float, dest='p')
parser.add_argument('-n',    help='Number of lines', required=False, type=int, dest='n')
args = parser.parse_args()


# methods

def seek_sampler(file, k):
    file.seek(0, 2)
    size = file.tell()
    lines = sorted(random.sample(range(size), k))

    for l in lines:
        file.seek(lines)
        file.readline()
        yield file.readline()

def reservoir_sampler(_file, k):
    sample = []
    for i, line in enumerate(_file):
        if i < k:
            sample.append(line)
        else:
            r = random.randint(0, i)
            if r < k:
                sample[r] = line
    return sample

def probability_sampler(_file,p):
    for line in _file:
        if random.random() < p:
            yield line

# running

sampler = None
file    = open(args.input, 'r')

if args.p:
    sampler = probability_sampler

if args.n:
    sampler = seek_sampler if args.input != '/dev/stdin' else reservoir_sampler

if sampler is None:
    print("You need to specify either -p (probability) or -n (sample size)")
    exit()

for line in sampler(file, args.p or args.n):
    print(line, end=' ')

