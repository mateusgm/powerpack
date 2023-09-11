#!/usr/bin/env pyspark

import os
import getpass
import sys

from pyspark.sql import SparkSession, Window, functions as F, types as T

input_path      = sys.argv[1]
output_path     = sys.argv[2]
partition_mb    = 1024
min_partitions  = 20

# spark

spark = SparkSession.builder.enableHiveSupport().getOrCreate()
log4j = spark.sparkContext._jvm.org.apache.log4j
logger = log4j.LogManager.getLogger(spark.sparkContext.appName)
logger.setLevel(log4j.Level.INFO)


# making output path sane

if output_path[:5] != '/user':
    output_path = '/user/{}/{}'.format( getpass.getuser(), output_path )
os.system( 'hdfs dfs -rm -r {}'.format(output_path) )

print "Saving TFRecords from {} in {}".format(input_path, output_path)


# number of partitions

def _dataframe_size_hdfs(path):
    if path[:5] != '/user':
        path   = '/user/hive/warehouse/{}'.format( path.replace('.', '.db/') )
    output = sb.check_output('hdfs dfs -du -s {}'.format(path, universal_newlines=True ), shell=True )
    return float(output.split(' ')[0]) / 1024 / 1024

df_size      = _dataframe_size_hdfs(input_path)
n_partitions = max(int(df_size / partition_mb), min_partitions)

spark \
    .table(input_path) \
    .repartition(n_partitions) \
    .write.format("tfrecords") \
    .save(output_path)
