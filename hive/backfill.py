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
import subprocess as sb
import datetime as dt
from multiprocessing.dummy import Pool as ThreadPool

templates = {
  'hive':   "beeline -f {} {} --hivevar DATE={}",
  'spark':  "spark-submit {} {} {}",
}

parser = argparse.ArgumentParser(description='Backfill a table in parallel without taking the oozie route')
parser.add_argument('begin_date')
parser.add_argument('end_date')
parser.add_argument('-a', dest='args')
parser.add_argument('-t', default=2,  dest='N_THREADS', type=int)
parser.add_argument('-i', default=1,  dest='increment', type=int)
parser.add_argument('-p', dest='fill_partitions')
for k in templates.keys():
  parser.add_argument("-{}".format(k))
args = parser.parse_args()


# scheduler

def get_dates():
    begin   = dt.datetime.strptime( args.begin_date, "%Y-%m-%d" )
    end     = dt.datetime.strptime( args.end_date,   "%Y-%m-%d" )
    ascendent = end > begin
    gradient  = 1 if ascendent else -1
    
    partitions = None
    if args.fill_partitions:
        path = "/user/hive/warehouse/{}.db/{}".format(*args.fill_partitions.split('.'))
        partitions = sb.check_output( "hadoop fs -ls " + path, shell=True , universal_newlines=True )

    dates = []
    while ( ascendent and end > begin ) or ( not ascendent and end < begin ):
        date = begin.strftime("%Y-%m-%d")
        begin = begin + dt.timedelta(days= gradient * args.increment)
        if not partitions or date not in partitions:
          dates.append(date)
    
    return dates

def process( date ):
    cmd = tmpl.format( vars(args)[mode], args.args or '', date )
    sb.call( cmd + " 2> ~/backfill.err > ~/backfill.out" , shell=True )


# running

mode = [ k for k,v in templates.items() if vars(args)[k] ][0]
if not mode:
    print("You need to specify either " + templates.keys())
    exit()

tmpl    = templates[ mode ]
dates   = get_dates()

pool    = ThreadPool( args.N_THREADS )
results = pool.imap_unordered( process, dates )

print("{} dates: {} ...".format( len(dates), dates ))
begin   = time.time()
for i, r in enumerate(results, 1):
    print("%d / %d :: %.1f m/job" % ( i, len(dates), float(time.time() - begin)/i/60 ))

pool.close()
pool.join()

