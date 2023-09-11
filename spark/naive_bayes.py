from math import log
from operator import add
from itertools import product
from collections import defaultdict
from itertools import combinations
from pyspark.sql import SparkSession, SQLContext
import numpy as np

# import naive_bayes as nb

# labels   = 'labels'

# df          = df.selectExpr('*', 'rand() < 0.95 as train').cache()
# train, test = df.where(df.train).cache(), df.where(~df.train).cache()

# predict, w = nb.train(df, labels=labels, features=[ 'ufi', 'site_type' ], spark=spark )
# best_fts   = nb.feature_selection(test, predict, labels=labels, weights=w, iterations=2)
# nb.export( w, predict, table='mmelo.app_popular_filters2', spark=spark, partitions=100 )


def train(df, labels='', features=None, smoothing=1.0, partitions=1000, spark=None):
    sc       = spark.sparkContext
    df       = df.cache()

    cols     = df.columns
    features = features or [ c for c in df.columns if c not in [ labels, 'train' ] ]

    # calculating count for (label, feature, value ) and (label)
    
    total_count   = df.count()
    
    label_count   = df.rdd \
        .flatMap(lambda x: [ (l, 1) for l in x[labels] ]) \
        .reduceByKey(add).collectAsMap()
    
    feature_count = df.rdd \
        .flatMap(lambda x: [ ( (l, f, x[f]), 1 ) for f in features for l in x[labels] ] ) \
        .reduceByKey(add).collectAsMap()


    # getting cardinality of each feature so we can do the smoothing

    feature_values = defaultdict(set)
    for (l,f,v),c in feature_count.iteritems():  feature_values[ f ].add( v )
    for f,vs      in feature_values.iteritems(): feature_values[ f ] = len( vs )


    # estimating likelihood

    likelihood_func = lambda l,f,c: log( float(c + smoothing) / ( label_count[l] + feature_values[f] ) )
    likelihood_map  = { (l,f,v): likelihood_func(l,f,count) for (l,f,v),count in feature_count.iteritems() }
    
    for l in label_count.keys():
        for f in feature_values.keys():
            likelihood_map[ (l,f,'unseen') ] = likelihood_func( l, f, 0 )
        likelihood_map[ (l,'prior','_') ] = log( float(label_count[l]) / total_count )


    # prediction function

    likelihood_brc = sc.broadcast(likelihood_map)

    def predict(row, _class, features=None, header=cols):
        score       = likelihood_brc.value.get((_class, 'prior', '_'))
        for f,v in zip(header,row):
            if features is not None and f not in features: continue
            score += likelihood_brc.value.get((_class,f,v)) or likelihood_brc.value.get((_class,f, 'unseen'))
        return score
    
    return predict, likelihood_map


def feature_selection(df, predict_func, labels=None, weights=None, early_stop=0.005, iterations=None ):
    cols        = df.columns
    label_i     = df.columns.index(labels)

    features    = set( f for _,f,_ in weights.keys() if f != 'prior')
    test_labels = set( l for l,_,_ in weights.keys() )

    def precision_at_k(row, features, thresholds=[1,3,5]):
        predictions = [ predict_func(row, l, features=features) for l in test_labels ]
        sorted_labl = zip(*sorted( zip(test_labels, predictions), key=lambda x: x[1], reverse=True))[0]
        truth_size  = len(row[label_i])
        return [ 1 ] + [ float(len(np.intersect1d(sorted_labl[:k], row[label_i]))) / min(truth_size, k) for k in thresholds ]

    def test_feature_sets(df, sets):
        results = df.rdd \
            .flatMap(lambda x: [ (f, precision_at_k(x, fts)) for f, fts in sets ]) \
            .reduceByKey(lambda a,b: [ a[i]+b[i] for i,_ in enumerate(a) ]) \
            .mapValues(lambda x: [ p / float(x[0]) for p in x[1:] ]) \
            .collectAsMap()
        return sorted(results.iteritems(), key=lambda x: x[1][-1], reverse=True)


    # stepwise forward selection

    to_consider = features
    log         = [ test_feature_sets(df, [ ('popularity', []) ])[0] ]
    best_fset   = []
    
    for i in range(iterations or len(features)):
        sets     = [ (f, best_fset + [f]) for f in to_consider ]
        scores   = test_feature_sets(df, sets)
        to_consider = zip(*scores)[0][1:]

        print "---------\n", "\n".join(map(str,scores))
        
        if iterations is None and scores[0][1][-1] - log[-1][1][-1] < early_stop: break
        best_fset.append( scores[0][0] )
        log.append( scores[0] )

    print "----------\n", "\n".join(map(str,log))
    return best_fset

def export(weights, predict, table=None, spark=None, partitions=500):

    fvalues  = defaultdict(lambda: defaultdict(set))
    for l,f,v in weights.keys():
        if f == 'prior' or v == 'unseen': continue
        fvalues[l][f].add((f,v))
    
    features = list(set(f for fts in fvalues.values() for f in fts.keys()))


    # doing combinations and its predictions

    def build_row(x):
        lookup = dict(x[1:])
        return [ lookup[f] for f in features ] + [ x[0], predict(lookup.values(), x[0], header=lookup.keys()) ] 
    
    sc   = spark.sparkContext
    data = sc.parallelize([])
    for l, fts in fvalues.iteritems():
        combinations = list( product(*([[l]] + fts.values())) )
        data         = data.union( sc.parallelize( combinations ).map(build_row) )


    # save to the table

    data \
        .toDF( list(features) + [ 'label', 'score' ] ) \
        .repartition( partitions ) \
        .registerTempTable('data')

    spark.sql("DROP TABLE IF EXISTS {}".format( table ))

    spark.sql("""
        CREATE TABLE {}
        ROW FORMAT DELIMITED FIELDS TERMINATED BY "\t" STORED AS TEXTFILE
        AS SELECT * FROM data
        ORDER BY {}, score DESC
    """.format( table, ','.join(features) ) )

    print "Exported", table

