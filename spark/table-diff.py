from pyspark.sql import functions as F

import sys
import pandas as pd
import argparse

# init

spark  = SparkSession.builder.enableHiveSupport().getOrCreate()
logger = spark.sparkContext._jvm.org.apache.log4j
level  = logger.Level.OFF
logger.LogManager.getLogger("org"). setLevel( level )
logger.LogManager.getLogger("akka").setLevel( level )

# params

parser = argparse.ArgumentParser()
parser.add_argument('table1',    help='Table 1')
parser.add_argument('table2',    help='Table 2')
parser.add_argument('columns',   help='Columns to be compared')
parser.add_argument('join_cols', help='Columns to join the tables')
parser.add_argument('partition', help='Date partition to filter the tables', default='')
args = parser.parse_args()
print args

cols      = args.columns.split(',')
join_keys = args.join_cols.split(',')

eps       = 3e-4
quantiles = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# helper

def load(table, sufix=''):
    df = spark.table( table )
    if args.partition and 'yyyymmdd' in df.columns:
        df = df.where(df.yyyymmdd == args.partition)
    final_cols = join_keys + [ F.col(c).alias(c+sufix) for c in df.columns if c in cols ]

    return df.select(*final_cols) \
        .repartition( 1000, *join_keys) \
        .cache()


# loading the data

df1 = load( args.table1, sufix='_1' )
df2 = load( args.table2, sufix='_2' )
df = df1.join(df2, join_keys).repartition( 1000, *join_keys).cache()

print df1.columns
print df2.columns
print (df1.count(), df2.count(), df.count())


# row by row bias

real_cols = []
for c in cols:
    if c+'_1' in df.columns and c+'_2' in df.columns:
        df = df.withColumn(c+'_diff', F.col(c+'_1') - F.col(c+'_2'))
        real_cols.extend([ c+'_1', c+'_2', c+'_diff' ])


# basic aggregations

metrics = { }
for c in real_cols:
    metrics['tiles_'+c] = df.approxQuantile( c, quantiles, eps )

cnt_value = lambda c,v: F.sum(F.expr("if({}={},1,0)".format(c,v))) / F.count(c) 

agg = df.groupBy().agg(*(
    [ F.count(c).alias('cnt_'+c)            for c in real_cols ] +
    [ cnt_value(c,0).alias('zeros_'+c)      for c in real_cols ] +
    [ F.stddev(c).alias('stddev_'+c)        for c in real_cols ] +
    [ F.min(c).alias('min_'+c)              for c in real_cols ] +
    [ F.max(c).alias('max_'+c)              for c in real_cols ] +
    [ F.mean(c).alias('mean_'+c)            for c in real_cols ] +
    [ F.sqrt(F.mean(F.pow(F.col(c),2))).alias('rmse_'+c) for c in real_cols if 'diff' in c ]
)).toPandas().squeeze()

for i,v in agg.items():
    metrics[i] = v

for c in real_cols:
    if 'diff' not in c:
        metrics['r2_'+c] =  1.0 - metrics['rmse_'+c[:-2]+'_diff']**2 / metrics['stddev_'+c]**2


# displaying

metrics_keys  = list({ m.split('_')[0] for m in metrics.keys() if 'tiles' not in m })
metrics_final = []

for c in sorted(real_cols):
    vals = [ ]
    for m in metrics_keys:
        vals.append(metrics[m+'_'+c] if m+'_'+c in metrics else None)
    for i,_ in enumerate(quantiles):
        vals.append( metrics['tiles_'+c][i] )
    metrics_final.append( vals )
metrics_keys.extend( [ 'quantile_'+str(q) for q in quantiles ] )

print pd.DataFrame(zip(*metrics_final), columns=sorted(real_cols), index=metrics_keys)


