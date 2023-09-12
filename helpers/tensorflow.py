from __future__ import absolute_import, print_function

import os
import time
import sys
import subprocess as sb
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

def setup_hdfs_env():
    path = sb.check_output('echo $LD_LIBRARY_PATH', shell=True, universal_newlines=True ).strip()
    os.environ['HADOOP_HDFS_HOME'] = "/usr/lib/hadoop-hdfs"
    os.environ['JAVA_HOME'] = '/usr/java/jdk1.8.0_144/jre'
    os.environ['HADOOP_HOME'] = '/usr/lib/hadoop'
    os.environ['HADOOP_HDFS_HOME'] = '/usr/lib/hadoop-hdfs'
    os.environ['LD_LIBRARY_PATH'] = ':'.join([path, os.environ['JAVA_HOME'] + '/lib/amd64/server'])
    os.environ['CLASSPATH'] = sb.check_output('/usr/lib/hadoop/bin/hadoop classpath --glob', shell=True, universal_newlines=True )

def input_fn(filenames, cols=[], x=None, y=None, batch_size=32, parallel=4, num_epochs=1):  
    def _parse_function(example_proto):  
        example    = tf.parse_example(example_proto, cols)
        example_x  = { c: f for c,f in list(example.items()) if c in x }
        return example_x, tf.reshape(example[y],(-1,1))
    
    def input_fn(): 
        files   = tf.data.Dataset.list_files(filenames)
        return tf.data.TFRecordDataset(files, buffer_size=None, num_parallel_reads=parallel) \
            .batch(batch_size) \
            .repeat(num_epochs) \
            .map(_parse_function, num_parallel_calls=parallel*2) \
            .prefetch( batch_size*parallel*2 ) # number of cpu
    return input_fn

def get_dataset(*args, **kwargs):
    return input_fn(*args, **kwargs)()

def get_iterator(*datasets, **kwargs):
    if 'feedable' in kwargs:
        handle    = tf.placeholder(tf.string, shape = [])
        it        = tf.data.Iterator.from_string_handle(handle, datasets[0].output_types, datasets[0].output_shapes)
        return it, handle, tuple([ d.make_initializable_iterator() for d in datasets ])

    else:
        it  = tf.data.Iterator.from_structure(datasets[0].output_types, datasets[0].output_shapes)
        return it, tuple( [ it.make_initializer(d) for d in datasets ] )

def run_epoch(sess, *args, **kwargs):
    try:
        vals = []
        before = time.time()
        while True:
            vals.append( sess.run(*args, **kwargs) )
            if len(vals) % 300 == 0: print('.', end=' ', file=sys.stderr)
    except tf.errors.OutOfRangeError:
        pass
    after  = time.time()
    print("\nEpoch done: %d batches / %.2f mins" % (len(vals), (after - before)/60), file=sys.stderr)
    return vals

def plot_learning_curve(*curves, **args):
    name = args['name']
    plt.figure(figsize=(16,9))

    for i, batches in enumerate(curves):
        label    = "{}.{}".format(name,i)
        cleaned  = np.array(batches)[~np.isnan(batches)]
        n        = len(cleaned)
        plt.plot( list(range(n)), cleaned, label=label )

        if 'rolling' in args:
            smoothed = pd.Series(cleaned).rolling(args['rolling']).mean() 
            plt.plot( list(range(n-len(smoothed), n)), smoothed, label=label+".smoothed" )

    plt.legend()
    plt.savefig(os.path.expanduser('~/plots/{}.png'.format(name)))
    plt.close()

