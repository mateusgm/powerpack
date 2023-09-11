#!/usr/bin/env python2
__author__ = 'lbernardi'

import os.path
import sys
import argparse
import gzip
import math
import re
import csv

parser = argparse.ArgumentParser(description='Vectorices a stream of examples. Outputs a lightsvm formatted file.')
parser.add_argument('-i', help='Input file. CSV. Default is standard input.', default='/dev/stdin', dest='input_file')
parser.add_argument('-iz', help='Input is gzipped.', dest='iz', action='store_true')
parser.add_argument('-o', help='Output file. Default is standard output.', default='/dev/stdout', dest='output_file')
parser.add_argument('-oz', help='Gzip output', dest='oz', action='store_true')
parser.add_argument('-sep', help='File column separator. Default: tab ', default='\t', dest='sep')
parser.add_argument('-r', help='No header',  dest='has_header', action='store_false')
parser.add_argument('-l', help='Label index. Last column by default.', dest='label_index', type=int, default=-1)
parser.add_argument('-c', help='Space separated list of categorical 0-based indices. By default all features are considered categorical,', dest='cats', default='')
parser.add_argument('-z', help='Read and write gzipped files', dest='gz', action='store_true')
parser.add_argument('-index_file', help='Output all feature values to a file sorted index in the sparse representation', dest='index_file')
parser.add_argument('-ixz', help='Gzip index file.', dest='ixz', action='store_true')
parser.add_argument('-chunk', help='Output manny files each with at most n lines, where is specified by this option.', dest='chunk_size', type=int)
parser.add_argument('-chunk_sufix_list', help='Comma separated ordered list of sufixes to be appended to each chunk file name.', dest='chunk_sufix_list')
parser.add_argument('-apply', help='Apply the given index file to the input.', dest='apply_file_name')

args = parser.parse_args()
input_file_name, output_file_name, sep, has_header, label_index, cats, index_file_name, chunk_size, gzipped_input, gzip_output, gzip_index, chunk_sufix_list, apply_file_name = args.input_file, args.output_file, args.sep, args.has_header, args.label_index, args.cats, args.index_file, args.chunk_size, args.iz, args.oz, args.ixz, args.chunk_sufix_list, args.apply_file_name

index_to_apply = None
if apply_file_name:
	if apply_file_name.endswith('.gz'):
                apply_file = gzip.open(apply_file_name)
	else:
		apply_file = open(apply_file_name)

	apply_csv = csv.reader(apply_file, delimiter=',')
	index_to_apply = {}
	for row in apply_csv:
		fn, fv, fi = row
		if fn not in index_to_apply:
			index_to_apply[fn]={}
		index_to_apply[fn][fv]=fi

if chunk_sufix_list:
    chunk_sufix_list = chunk_sufix_list.strip().split(',')

categorical_features_indices = []
if cats !='':
    categorical_features_indices = map(int, cats.strip().split())



categorical_features_values = {}
numeric_features_indices = None

if gzipped_input:
    input = gzip.open(input_file_name)
else:
    input = open(input_file_name)

inputcsv = csv.reader(input, delimiter=sep)

header = None
if has_header:
    header = inputcsv.next()

if args.index_file:
    if gzip_index:
        index_file = gzip.open(index_file_name, 'w')
    else:
        index_file = open(index_file_name, 'w')

if args.chunk_size:
    current_chunk = 0
    filename, file_extension = output_file_name[:output_file_name.find('.')], output_file_name[output_file_name.find('.'):]
    sufix = str(current_chunk) if not chunk_sufix_list else chunk_sufix_list[current_chunk]
    if gzip_output:
        output = gzip.open(filename+'_'+sufix+file_extension, 'w')
    else:
        output = open(filename+'_'+sufix+file_extension, 'w')
else:
    if gzip_output:
        output = gzip.open(output_file_name, 'w')
    else:
        output = open(output_file_name, 'w')

if index_to_apply:
	for i, row in enumerate(inputcsv):
		if i==0:
			total_cols = len(row)
	        	if label_index<0:
        	        	label_index = total_cols+label_index
		line = row[label_index]
		for j, fvs in enumerate(row):
			if j != label_index:
				fn_index = index_to_apply[header[j]]
				for fv in fvs.split(' '):
                                        try:
                                                hot_index = fn_index['*']
                                        except KeyError:
                                                pass

					try:
						hot_index = fn_index[fv]
					except KeyError:
						continue
					line += ' %s:1' % (hot_index)
		line+='\n'
		output.write(line)
	output.close()
	sys.exit()

for i, row in enumerate(inputcsv):
    if i==0:
        total_cols = len(row)
	categorical_features_indices = range(total_cols)
	if label_index<0:
		label_index = total_cols+label_index

        categorical_features_indices.remove(label_index)

        numeric_features_indices = []
        for j in range(total_cols):
            if not (j in categorical_features_indices) and j !=label_index:
                numeric_features_indices.append(j)
        max_index_so_far = len(numeric_features_indices)
        if args.index_file:
            for j, k in enumerate(numeric_features_indices):
                if header:
                    k = header[k]
                    index_file.write("%s,%s,%s\n"%(k, '*', j))

    if args.chunk_size and i>0 and i % chunk_size == 0:
        output.close()
        current_chunk +=1
 	filename, file_extension = output_file_name[:output_file_name.find('.')], output_file_name[output_file_name.find('.'):]
    	sufix = str(current_chunk) if not chunk_sufix_list else chunk_sufix_list[current_chunk]
        if gzip_output:
            output = gzip.open(filename+'_'+sufix+file_extension, 'w')
        else:
            output = open(filename+'_'+sufix+file_extension, 'w')

    if len(row)!=total_cols:
        print line, len(row)
	continue
    label = row[label_index]
    line = label + ' '

    # First positions of the vectorized example are allocated for the Numerical features
    #++++++++++++++++++++++++++++++++++++++
    for j, k in enumerate(numeric_features_indices):
        line += '%s:%s ' % (j, row[k].strip())
    #++++++++++++++++++++++++++++++++++++++

    hot_indices = []
    for j in categorical_features_indices:
	
        feature_values = row[j].strip().split(' ')
	for _, feature_value in enumerate(feature_values):
	        try:
	            hot_index = categorical_features_values[(j, feature_value)]
	        except KeyError:
	            categorical_features_values[(j, feature_value)] = max_index_so_far
	            hot_index = max_index_so_far
	            if args.index_file:
	                if header:
	                    index_file.write("%s,%s,%s\n"%(header[j], feature_value, max_index_so_far))
	                else:
	                    index_file.write("%s,%s,%s\n"%(j, feature_value, max_index_so_far))
	            max_index_so_far+=1
	        hot_indices.append(hot_index)

    for h in hot_indices:
        line += '%s:1 '%h

    line = line.strip()
    output.write(line+'\n')

output.close()
