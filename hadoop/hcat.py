#!/usr/bin/env python3
# vim: syntax=python
#
# usage:
#   hcat.py  mydb.mytable > table.tsv
#
# to save partitioned gz files:
#   hcat.py mydb.mytable -p /tmp/dataset
#
# to stream files through a mapper in parallel:
#   hcat.py mydb.mytable -m my_parser.py
#
from __future__ import print_function

import argparse
import os
import sys
import random
import tempfile
import subprocess as sb
import multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument('path', help='Hive table or hdfs path')
parser.add_argument('-m',         dest='mapper',            default='',         help='Mapper script that the data will be streamed through in parallel')
parser.add_argument('-p',         dest='partitions_path',   default='',         help='Destination folder to save the compressed partitions')
parser.add_argument('-t',         dest='n_threads',         default=0,          type=int, help='Number of threads')
parser.add_argument('-b',         dest='batch_size',        default=1,          type=int, help='Batch size - number of hdfs files copied per thread')
parser.add_argument('-s',         dest='sample',            default=None,       type=float, help='Sample probability with which to sample the hdfs files')
parser.add_argument('-c',         dest='files_cache',       default=None,       help='Cache path for list of hdfs paths, useful if too many partitions')
parser.add_argument('--header',   dest='header',            default='',         help='Mimic header from this table')
parser.add_argument('--hive_dir', dest='hive_dir',          default='/user/hive/warehouse/', help='Base dir for hive tables')
parser.add_argument('--eta',      dest='print_eta',         action='store_true', help='Print ETA while downloading')
parser.add_argument('--fast',     dest='fast',              action='store_true', help='Enable flags to reduce friction: --noheader and --nochars.')
parser.add_argument('--noheader', dest='noheader',          action='store_true', help='Do not print the header of the table')
parser.add_argument('--nochars',  dest='nochars',           action='store_true', help='Do not try to deal with hadoop special chars. Use only if you created the table with \'ROW FORMAT DELIMITED FIELDS TERMINATED BY "\\t"\' and you dont have any complex types (arrays and maps).')
parser.add_argument('--cmd',      dest='cmd',               action='store_true', help='Do not run the command, only print it. Useful if you want to stream data through ssh and parallel is dropping some chars')
parser.add_argument('--verbose',  dest='verbose',           action='store_true', help='Print informational messages')
args = parser.parse_args()

cmds        = []
output_cmds = ''

if args.fast:
    args.noheader = 1
    args.nochars = 1


# helpers

def info(*msg):
    if args.verbose:
        print(">", *msg, file=sys.stderr)
        sys.stderr.flush()

def get_path(path):
    if path[:5] in ('gs://', 'hdfs:', '/user'):
        return path

    if args.hive_dir:
        return args.hive_dir + path.replace('.', '.db/')

    out = hive('describe formatted '+path, pipe=' | grep Location')
    return out.strip().split("\t")[-2]

def get_header(table):
    description = hive('describe '+table)
    features    = [ ]
    for l in description.split("\n")[1:]:
        definition = [ x.strip(" \t'") for x in l.split("\t")[:2] ]
        if definition[0] == '': continue
        if '#' == l[0]:         break
        features.append(definition)
    return zip(*features)

def hive(query, pipe=''):
    cmd = "beeline --silent=true --showHeader=true --incremental=true --outputformat=tsv2 -e '{}' 2> /dev/null".format(query)
    return run(cmd + pipe)

def run(*args):
    return sb.check_output(' '.join(args), shell=True, universal_newlines=True)


# getting files

is_table = False
path     = get_path( args.path )
if path != args.path:
    info( "Relative path detected - assuming hive table" ) 
    is_table = True

try:
    if args.files_cache and os.path.exists(args.files_cache):
        files = open(args.files_cache, 'r').read().strip().split("\n")
    else:
        exclude = [ 'SUCCESS', '/_temporary', '/.hive-staging' ]
        files = run("hdfs dfs -ls -C", path).split("\n")
        files = [ f for f in files if f.strip() and sum([ int(e in f) for e in exclude ]) == 0 ]
    info(len(files), "files found")

except Exception as e:
    info( "Error getting files\n Does the table/hdfs_path exists?" )
    exit()

if len(files) == 0:
    info( "No files found. Does the table has any data?" )
    exit()

if args.sample:
    sample_size = max(1,int(round(args.sample*len(files))))
    files = random.sample(files, sample_size) 
    info(len(files), "files sampled")

# composing command: cat

cmds = [ 'hdfs dfs -cat {}' ]

if '=' in files[0].split('/')[-1]:
    files = [ f+'/*' for f in files ]

if '.gz' in files[0] and not args.partitions_path:
    info( "Detected gzip files" )
    cmds.append( 'zcat' )

if '.bz' in files[0]:
    info( "Detected bzip files" )
    cmds.append( 'bzcat' )


# writing file queue

_, input_path = tempfile.mkstemp()
if args.files_cache:
    input_path = args.files_cache

with open(input_path, 'w') as tmp:
    tmp.write("\n".join(files))


# composing command: special chars

if not args.nochars:
    special = {
        r'\x00': '\\n',
        r'\x01': '\\t',
        r'\x02': ',',
        r'\x03': ':',
        r'\\\N': 'NULL',

    }

    replaces = [ "s/%s/%s/g;" % s  for s in special.items() ]
    sed_cmd  = "sed -e \"{}\"".format( " ".join(replaces) ) 
    cmds.append( sed_cmd )


# getting header

if (is_table and not args.noheader) or args.header:
    info( "Getting header" )
    header, _ = get_header( args.header or args.path )
    if args.partitions_path or args.mapper:
        cmds[0] = "echo -e \"{}\"; {}".format( '\\t'.join(header), cmds[0])
    else:
        print("\t".join(header))
        sys.stdout.flush()

# mappers

if args.mapper:
    info( "Mapping through every row" )
    output_cmds +=  " | %s " % (args.mapper )


# saving to partitions

if args.partitions_path:
    info( "Partitions respected and saved to", args.partitions_path )
    os.system( "mkdir -p {}".format( args.partitions_path ) )
    path_placeholder = args.partitions_path.rstrip('/') + "/{1/.}.gz"
    compress = '' if '.gz' in files[0] else '| pigz'
    output_cmds += "{c} > {p}; echo {p}".format(c=compress, p=path_placeholder)


# final command

eta       = "--eta" if args.print_eta else ""
threads   = args.n_threads  or min(mp.cpu_count()-2, 4)

inner_cmd = ' | '.join(cmds)
if len(output_cmds) > 0:
    inner_cmd = "( {} ) {}".format( inner_cmd, output_cmds )

final_cmd = "cat {files} | T={threads} N={batch} parallel.sh {cmd}".format( \
        threads=threads, \
        batch=args.batch_size, \
        eta=eta, \
        cmd=inner_cmd, \
        files=input_path )


# copying in parallel

info( "Starting to copy in parallel ({} threads)".format(threads) )
if args.cmd:
    print(final_cmd)
    exit()

try:
    sb.call( final_cmd, shell=True )
except KeyboardInterrupt:
    pass
except BrokenPipeError:
    pass

