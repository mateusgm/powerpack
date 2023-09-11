from __future__ import absolute_import, print_function
import subprocess as sb

from pyspark.sql import functions as F, types as T, SparkSession
from pyspark.sql.window import Window

# usage:
#
# import os
# import spark_helper as hlp
# spark, sc   = hlp.init_spark()
# hlp.supress_log()
#

# meta helpers

def my_spark():
    return SparkSession.builder.enableHiveSupport().getOrCreate()

def init_spark(logging_off=True):
    spark  = my_spark()
    monitoring_url()
    supress_log(logging_off)
    return spark, spark.sparkContext

def monitoring_url():
    spark     = my_spark()
    proxy_uri = 'spark.org.apache.hadoop.yarn.server.webproxy.amfilter.AmIpFilter.param.PROXY_URI_BASES'
    url       = spark.sparkContext._conf.get( proxy_uri )
    if url: print("Monitoring URL: " + url.split(',')[0])

def supress_log(logging_off=True):
    spark  = my_spark()
    logger = spark.sparkContext._jvm.org.apache.log4j
    level  = logger.Level.OFF if logging_off else logger.Level.ERROR
    logger.LogManager.getLogger("org"). setLevel( level )
    logger.LogManager.getLogger("akka").setLevel( level )


# algorithms

def backfill(callback, _since, _until, step=1):
    import datetime as dt
    import time
    current = dt.datetime.strptime( _since, "%Y-%m-%d" )
    finish  = dt.datetime.strptime( _until, "%Y-%m-%d" )
    reverse = current > finish
    while (reverse and current > finish) or current < finish:
        callback( current )
        current = current + ( -1 if reverse else 1 ) * dt.timedelta(days=STEP)


# data helpers

def export_table(name, df):
    df.registerTempTable('df')
    my_spark().sql("DROP TABLE IF EXISTS {}".format(name))
    my_spark().sql("""
        CREATE TABLE {}
        ROW FORMAT DELIMITED FIELDS TERMINATED BY "\t" STORED AS TEXTFILE
        AS SELECT * FROM df
    """.format(name))

def checkpoint(name, df, force=False, partitions=None, user='mmelo'):
    path = "hdfs:///user/{}/ckps/{}".format(user, name)
    try:
        if not force:
            return my_spark().read.parquet(path)
    except Exception as e:
        print(e)
        pass
    df.write.parquet(path, mode='overwrite')
    return my_spark().read.parquet(path)


def _dataframe_size_java(df):  
    from pyspark.serializers import PickleSerializer, AutoBatchedSerializer
    rdd    = df.rdd._reserialize(AutoBatchedSerializer(PickleSerializer()))
    obj    = rdd.ctx._jvm.org.apache.spark.mllib.api.python.SerDe.pythonToJava(rdd._jrdd, True)
    nbytes = my_spark().sparkContext._jvm.org.apache.spark.util.SizeEstimator.estimate(obj)
    return nbytes/1024/1024

def _dataframe_size_hdfs(path):
    if path[:5] != '/user':
        path   = '/user/hive/warehouse/{}'.format( path.replace('.', '.db/') )
    output = sb.check_output('hdfs dfs -du -s {}'.format(path, universal_newlines=True ), shell=True )
    return float(output.split(' ')[0]) / 1024 / 1024

def dataframe_size(target, java=False):
    if java:
        return _dataframe_size_java(target)
    return _dataframe_size_hdfs(target)

def n_partitions(input_path, size=1024, min_partitions=20, java=False):
    df_size = dataframe_size(input_path, java=java)
    return max(int(df_size / size), min_partitions)

def time_travel(on, over=None, window=None):
    from pyspark.sql.window import Window
    return Window.partitionBy(over).orderBy(on) \
        .rangeBetween(window[0]*86400, window[1]*86400)

def profile_spark(func):
    stageMetrics = sc._jvm.ch.cern.sparkmeasure.StageMetrics(my_spark()._jsparkSession)
    stageMetrics.begin();
    print("-------> Result:", func())
    stageMetrics.end()
    stageMetrics.printReport()

