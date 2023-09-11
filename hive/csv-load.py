#!/usr/bin/env python3
from __future__ import print_function

import sys
import subprocess as sb
import pandas as pd
import numpy as np
import getpass
import argparse

parser = argparse.ArgumentParser(description='Naive csv to hive importer')
parser.add_argument('-i', dest='csv_path', help='Path to the csv', required=True)
parser.add_argument('-t', dest='table',    help='Destination table', required=True)
parser.add_argument('-s', dest='sep',      help='Separator', default=",")
args = parser.parse_args()

type_lkp = {
    int: 'int',
    np.dtype('int32'): 'int',
    np.dtype('int64'): 'bigint',
    np.dtype('object'): 'string',
    np.dtype('float64'): 'float',
    np.dtype('float32'): 'float',
    float: 'float',
    bool: 'boolean',
}

# hdfs

print("Uploading csv", file=sys.stderr)

hdfs_path   = '/user/{}/{}'.format( getpass.getuser(), args.csv_path.replace('/', '_') )

sb.call( 'tail -n +2 {f} > {f}.noheader'.format(f=args.csv_path), shell=True )
sb.call( 'hadoop fs -rm -f {}'.format(hdfs_path), shell=True )
sb.call( 'hadoop fs -put {} {}'.format(args.csv_path + '.noheader', hdfs_path), shell=True )


# hive

df    = pd.read_csv(args.csv_path, sep=args.sep, nrows=100)
types = [ "`%s` %s" % (c,type_lkp[t]) for c,t in zip(df.columns, df.dtypes) ]

hive  = """
  DROP TABLE IF EXISTS {table};
  CREATE TABLE {table} ( {types} )
  ROW FORMAT DELIMITED FIELDS TERMINATED BY "{sep}" STORED AS TEXTFILE;
  LOAD DATA INPATH "{hdfs}" OVERWRITE INTO TABLE {table};
""".format(table=args.table, types=','.join(types), hdfs=hdfs_path, sep=repr(args.sep))
cmd  = "beeline -e '{}'".format(hive.replace("\n", ' '))

print("Creating table", file=sys.stderr)
print(cmd, file=sys.stderr)

sb.call( cmd,  shell=True )
