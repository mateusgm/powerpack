#!/usr/bin/env python3
# vim: syntax=python
#
# usage:
#    backfill <script> <date-from> <date-to> "<params>"
#
from __future__ import print_function

import sys
import time
import argparse
import os
import subprocess as sb

parser = argparse.ArgumentParser()
parser.add_argument('mapper', help='Map command')
parser.add_argument('-i',     help='Input: hive table, HDFS directory or joblist',           dest='input')
parser.add_argument('-l',     help='Input: treat it as joblist, process N lines per mapper', dest='mapper_input', default=False)
parser.add_argument('-o',     help='Output HDFS directory',            dest='output')
parser.add_argument('-f',     help='Files to submit on the job',       dest='files')
parser.add_argument('-v',     help='VCores per mapper. Default: 1',    dest='vcores', default=1)
parser.add_argument('-m',     help='Memory per mapper. Default: 4096', dest='memory', default=4096, type=int)
parser.add_argument('-c',     help='Output codec',                     dest='codec', choices=['Gzip', 'BZip2', 'Deflate', 'Snappy'], default=False)
parser.add_argument('-p',     help='Enable preemption. Default: 1',    dest='preemption', default=True)
args = vars(parser.parse_args())

def call(cmd):
    print("Executing", cmd, file=sys.stderr)
    proc   = sb.Popen( cmd, shell=True, stdout=sb.PIPE, universal_newlines=True )
    for line in iter(proc.stdout.readline, ''):
        print("_", line)

def get_path(path):
    if path[:5] in ('gs://', 'hdfs:', '/user'):
        return path
    return "/user/hive/warehouse/" + path.replace('.', '.db/')

# options

args['properties']    = ''
args['input_format']  = ''
args['output_format'] = ''
args['output_codec']  = ''
args['files_to_upload']  = ''

if args['mapper_input']:
    args['input_format']  = "-inputformat  org.apache.hadoop.mapred.lib.NLineInputFormat"
    args['properties']   += " -D mapreduce.input.lineinputformat.linespermap={} ".format(args['mapper_input'])
    if args['input'][:5] != '/user':
        call( "hadoop fs -rm  -f {}".format( args['input'], args['input']) )
        call( "hadoop fs -put {} {}".format( args['input'], args['input']) )
    
else:
    args['input']  = get_path( args['input'] )
    args['output'] = args['output']

if args['codec']:
    allowed = [ 'Gzip', 'BZip2', 'Deflate', 'Snappy' ]
    if args['codec'] not in allowed:
        print("Error: codec should be one of", allowed, type=sys.stderr)
        exit()

    args['properties'] += """\
        -D mapreduce.output.fileoutputformat.compress=true \
        -D mapreduce.output.fileoutputformat.compress.codec=org.apache.hadoop.io.compress.{}Codec \
    """.format( args['codec'] )

if args['files']:
    args['files_to_upload'] = '-files ' + args['files']

elif os.path.exists(args['mapper'].split(' ')[0]):
    args['files_to_upload'] = '-files ' + args['mapper']

# if args['format']:
    # -D stream.reduce.output=text
    # -inputformat org.apache.hadoop.hive.ql.io.orc.OrcInputFormat \


# cmd

cmd = """
    hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar {properties} \
        -D yarn.resourcemanager.scheduler.monitor.enable={preemption} \
        -D mapreduce.task.timeout=600000000 \
        -D mapreduce.map.memory.mb={memory} \
        -D mapreduce.job.reduce=0 \
        -D mapreduce.map.cpu.vcores={vcores} \
        {files_to_upload} -input {input} \
        -output {output} \
        -mapper "{mapper}" \
        -reducer NONE \
        {input_format} {output_format}
""".format(**args)


call( "hadoop fs -rm -r -f " + args['output'] )
call( cmd )


