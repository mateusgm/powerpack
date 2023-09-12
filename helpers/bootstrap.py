
from pyspark.sql import SparkSession, functions as F, types as T
from operator import add, mul
from collections import defaultdict

import scipy.stats as st
import numpy as np
import os
import binascii


# rdd api

_slice    = lambda x,c: tuple([ x[i] for i in c ])
sum_items = lambda a,b: map(lambda i,j: map(add, i, j), a, b)
cartesian = lambda a,b: map(lambda i: map(mul, a, [i]*len(a)),b)
seed      = lambda: int(binascii.hexlify(os.urandom(4)), 16)
poisson   = lambda mu, size: np.random.RandomState(seed()).poisson(mu, size=size).tolist()

def bootstrap(df, metrics=[], post_metrics=[], group_by=[], n=1000):
    obs = df.where( ' AND '.join([ 'coalesce(%s,0) != 0' % m for m in metrics ]) )
    pop = df.groupBy(*group_by).count().orderBy(*group_by).collect()
    smp = obs.groupBy(*group_by).count().orderBy(*group_by).collect()

    cols  = metrics + [ 'pow(%s,2) as %s_var' % (m,m) for m in metrics ] + [ '1 as count' ]  
    names = [ s.split(' as ')[-1] for s in cols ] 
    zeros = [ [0]*len(cols) for i in range(n) ]  

    stats = obs \
        .selectExpr(*cols+group_by) \
        .rdd.map(lambda x: (_slice(x,group_by), cartesian(_slice(x, names), poisson(1, n)))) \
        .foldByKey(zeros, sum_items) \
        .sortByKey() \
        .collect()

    for i,(k,s) in enumerate(stats):
        new_s = np.array(s).T
        cnt   = new_s[-1,:] + poisson(pop[i][0]-smp[i][0], n)
        stats[i] = dict(zip(names, np.vstack([ new_s[:-1,:]/cnt, cnt])))
        for m in metrics:
            stats[i][m+'_var'] -= stats[i][m]**2
        for m,f in post_metrics.items():
            stats[i][m] = f(stats[i])

    return stats


# confidence intervals

def c_interval(values, ci_alpha=None, ci_method=None):
    ci_alpha  = ci_alpha or 90
    ci_method = ci_method or 'basic'
    lower, upper = (100-ci_alpha)/2.0, (100+ci_alpha)/2.0

    if ci_method == 'percentile':
        bounds = lambda m: np.percentile(m, [lower, upper ])
    
    if ci_method == 'basic':
        bounds = lambda m: 2*np.mean(m) - np.percentile(m, [ upper, lower ])

    if ci_method == 'normal':
        bounds = lambda m: st.norm(*st.norm.fit(m)).interval(ci_alpha/100.0)

    return bounds(values)

def diff_means_ci(stats, ci_alpha=None, ci_method=None):
    res = {}
    for m in stats[0].keys():
        if m[-2:] != '_e' and m[-4:] != '_var' and m != 'count':
            diffs  = np.array(stats[1][m]) - np.array(stats[0][m])
            res[m] = {
                'mean': np.mean(diffs),
                'ci': c_interval(diffs, ci_alpha=ci_alpha, ci_method=ci_method)
            }
    return res


# tests

tstat   = lambda m,v,a,b: ( b[m] - a[m] ) / np.sqrt( a[v]/a['count'] + b[v]/b['count'] )
t_comp  = lambda m,a,b,a0,b0: tstat(m+'_e', m+'_e_var', a, b) >= tstat(m+'_avg', m+'_var', a0, b0)
dictzip = lambda d: [ dict(zip(d,v)) for v in zip(*d.values()) ]

def efron_test(df, **kwargs):
    kwargs  = defaultdict(lambda: None, kwargs)
    metrics = kwargs['metrics']
    kwargs['metrics'] = metrics + [ m+'_e' for m in metrics ]
    
    prior, df = __efron_transform(df, metrics=metrics, group_by=kwargs['group_by'])
    a0, b0 = prior.orderBy(kwargs['group_by'][0]).collect()

    stats = bootstrap(df, **kwargs)
    sdict = map(dictzip, stats)
 
    res = diff_means_ci(stats, ci_method=kwargs['ci_method'], ci_alpha=kwargs['ci_alpha'])
    for m in res.keys():
        res[m]['p'] = None
        if m + '_avg' in a0: 
            res[m]['p'] = np.mean([ t_comp(m, sdict[0][i], sdict[1][i], a0, b0) for i in range(kwargs['n']) ])

    return res, stats

def cdf_test(df, stats=None, **kwargs):
    kwargs = defaultdict(lambda: None, kwargs)
    stats  = stats or bootstrap(df, **kwargs)
    res    = diff_means_ci(stats, ci_method=kwargs['ci_method'], ci_alpha=kwargs['ci_alpha'])
    for m in res.keys():
        diff        = np.array(stats[1][m]) - np.array(stats[0][m])
        res[m]['p'] = 2*min([ np.mean(diff < 0), np.mean(diff > 0) ])
    return res, stats

def z_test(df, stats=None, **kwargs):
    kwargs = defaultdict(lambda: None, kwargs)
    stats  = stats or bootstrap(df, **kwargs)
    res    = diff_means_ci(stats, ci_method=kwargs['ci_method'], ci_alpha=kwargs['ci_alpha'])
    for m in res.keys():
        diff        = np.array(stats[1][m]) - np.array(stats[0][m])
        mu, std     = st.norm.fit(diff)
        res[m]['p'] = 2*(1 - st.norm(0,std).cdf(abs(mu)) )
    return res, stats


def __efron_transform(df, metrics=[], group_by=[]):
    prior = df.groupBy(*group_by) \
        .agg(*[ F.count('*').alias('count') ] \
            + [ F.mean(F.coalesce(m,F.lit(0))).alias(m+'_avg')     for m in metrics ] \
            + [ F.variance(F.coalesce(m, F.lit(0))).alias(m+'_var') for m in metrics ]) \
        .cache()

    z = df.groupBy() \
        .agg(*[ F.mean(m).alias(m) for m in metrics ]) \
        .collect()[0]

    df = df.join( F.broadcast(prior), group_by )
    for m in metrics:
        df = df.withColumn(m+'_e', df[m] - df[m+'_avg'] + F.lit(z[m]))

    return prior, df

