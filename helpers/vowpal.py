from __future__ import absolute_import, print_function

import os
import sys
import re
import math
import random
import tempfile

from collections import defaultdict

NULLS           = (None, 'NULL', '\\N', '', 'NaN', 'undef')
NULL_VALUE      = 'NULL'
blocked_chars   = (' ', '|', ':', "\n")

prohibited_chars = r'[:|\s=^*%]'
max_digits      = 6
aliases         = [ 'label', 'weight', 'tag' ]
verbose         = False


# types

class raw_type():
    sep  = None
    null = ''

    @staticmethod
    def func(ft, params):
        if len(params) > 0:
            return lambda x: x.replace(params[0], ' ')
        return lambda x: x

class num_type():
    sep  = ':'
    null = 0

    @staticmethod
    def func(ft, params):
        return lambda x: x

class cat_type():
    sep  = '='
    null = NULL_VALUE

    @staticmethod
    def func(ft, params):
        return lambda x: str(x).replace(' ', '__')

class list_type(raw_type):
    @staticmethod
    def null(ft):
        return list_type.wrap_list(ft)( NULL_VALUE )

    @staticmethod
    def wrap_list(ft, vw_sep=cat_type.sep, list_sep=','):
        tmpl = "%s "
        if vw_sep.strip():
            tmpl = "{}{}%s ".format(ft, vw_sep)

        def wrapper(x):
            values = x.replace(' ', '__').split(list_sep)
            return (tmpl*len(values)) % tuple(values)
        return wrapper

    @staticmethod
    def func(ft, params):
        return list_type.wrap_list( ft )

# labels

class logistic_func(num_type):
    null = NULL_VALUE

    @staticmethod
    def func(ft, params):
        return lambda x: -1 if float(x) <= 0 else 1

class split_func(raw_type):

    @staticmethod
    def func(ft, params):
        return lambda x: x.replace(',', ' ')

class classweight_func(num_type):
    null = NULL_VALUE

    @staticmethod
    def func(ft, params):
        lookup = defaultdict(lambda: 1, [ tuple(p.split(':')) for p in params ])
        return lambda x: lookup[x]


# transformations

class int_func(cat_type):

    @staticmethod
    def func(ft, params):
        return lambda x: int(float(x))

class bool_func(cat_type):

    @staticmethod
    def func(ft, params):
        return  lambda x: int(bool(_to_float(x)))


class clip_func(num_type):

    @staticmethod
    def func(ft, params):
        bounds = list(map(float,params))
        cb     = lambda x: max( min( bounds[1], float(x) ), bounds[0] )
        return cb

class math_func(num_type):

    @staticmethod
    def _get_func(f, *modules):
        for m in modules:
            try:
                if hasattr(m,f):
                    return getattr(m,f)
                return globals()['__builtins__'][f]
            except:
                pass
        return

    @staticmethod
    def func(ft, params):
        func = math_func._get_func(params[0], math)
        _assert(func is not None, params[0] + ': coulnt find function ')
        cb   = lambda x: _try(ft, func, float(x), *tuple(map(int,params[1:])))
        return cb

class bin_func(cat_type):

    @staticmethod
    def _digitize_n(vals, bins, unique=False):
        results = set()
        highest = len(bins)-1
        for i,n in enumerate(map(float,vals)):
            j = highest
            while j > -1 and n < bins[j]: j -= 1 
            results.add( str(j+1) )
        return list(results)

    @staticmethod
    def _digitize_1(value, bins):
        highest = len(bins)-1
        while highest > -1 and value < bins[highest]:
            highest -= 1 
        return highest+1

    @staticmethod
    def func(ft, params):
        if 'exp' in params:
            return lambda x: int(math.log(float(x),2)) + 1 
        bins = list(map(float,params))
        return lambda x: bin_func._digitize_1(float(x), bins)

    # if t == 'escaped':
        # sep = None
        # cb  = lambda x: re.sub(r'[:|]', '*', x).split(' ')
        # _type = list

    # if 'likelihood' in t:
        # sep=None
        # cb = lambda f,x: ''.join((f, " ", f, "=", x.replace(vw_sep, '__')))
        # allow_nulls = True
    
    # if 'lookup' in t:
        # key = t.split(' ')[-1]
        # lkp = config[key]
        # cb  = lambda x: str(lkp[x])


def transformer(ft, desc, config=None, memoize=True):
    func    = None
    grammar = '^(cat|num|raw|list)? *([^ ]*)( .+)?$'

    _type, op, params  = re.findall(grammar, desc)[0]
    params = [ p for p in re.split(' |,', params.strip()) if p ]

    try:
        if op:
            if _type == 'num' and op+'_func' not in globals():
                params = [ op ] + params
                op = 'math'

            if op+'_func' in globals():
                claz = globals()[ op+'_func' ]
                sep  = claz.sep
                null = claz.null
                func = claz.func(ft, params)

        if _type:
            claz = globals()[ _type+'_type' ]
            sep  = claz.sep
            null = claz.null
            func = func or claz.func(ft, params)

        _assert( func is not None, op or _type )

    except RuntimeError as e:
        _exception( "Function '{}' not found  (used in '{}: {}')".format( e.message, ft, desc ) )


    # dealing with nulls

    if callable(null):
        null = null(ft)

    if memoize:
        promise, cache = _memoize(func)
        for n in NULLS:
            cache(n, null)
        return promise, sep

    return lambda x: null if x in NULLS else func(x), sep

# helpers

def _memoize(f):
    class memodict(dict):
        __slots__ = ()
        def __missing__(self, key):
            self[key] = ret = f(key)
            return ret
    cache = memodict()
    return cache.__getitem__, cache.__setitem__

def _assert(exp, message):
    if not exp:
        _exception(message)

def _exception(message):
    print("RuntimeError: \n    ", message, file=sys.stderr)
    exit(1)

def _to_float(x):
    try:
        return float(x)
    except:
        return x

def _try(ft, func, *args):
    try:
        return func(*args)
    except:
        _exception("{} | cant apply '{}' on ({})".format( ft, func.__name__, *args ))


# feature definition

def get_feature_config(config, exp_names=[], header=[], indexes=False, compress=False, memoize=True):
    config      = _get_config_with_defaults( config, exp_names )
    ns_lookup   = _get_namespaces(header, config)
    ft_lookup   = dict(_columns_iterator(header, config, with_aliases=True, memoize=memoize))

    feature_list = [ ]
    vw_template  = [ ]
    try:
        def add_token(f, name=None):
            col, func, sep = ft_lookup[f]
            index = header.index(col) if indexes else col
            feature_list.append( (index, func) )
            vw_template.append( "{}{}%s".format(name if sep and name else '', sep or '') )

        for a in aliases:
            if a in config['aliases'] and config['aliases'][a]:
                add_token( a )

        last_id = 1
        for i, (ns,fts) in enumerate(sorted(ns_lookup.items())):
            vw_template.append( "|{}".format(ns) )
            for f in fts:
                add_token( f, last_id if compress else f )
                last_id += 1

        return feature_list, ' '.join(vw_template).strip()

    except ValueError as e:
        col = re.findall("'.*'", e.args[0])[0]
        _exception("Couldnt find {} on the header".format(col))

def _get_config_with_defaults(config, exp_names):
    config  = defaultdict(dict, config)

    for e in exp_names:
        if verbose:
            print("Applying experiment:", e, file=sys.stderr)

        for k,v in config[e].items():
            if k in aliases:           config['aliases'][k] = v
            elif v:                    config['features'][k] = v
            else:                      config['features'].pop(k, None)

    default_a = dict(
        label_col= config['aliases']['label'].split(' ')[0],
        tag=       config['aliases']['label'],
        weight=    '', )

    for a,v in default_a.items():
        if a not in config['aliases'] or not config['aliases'][a]:
            config['aliases'][a] = v
   
    return config

def _get_namespaces(header, config, order=None):
    ns_desc  = defaultdict(str, config['namespaces'])
    ns_tree  = defaultdict(list)
    ns_chars = _possible_chars(ns_desc)

    features = list(_columns_iterator(header, config, order=order))
    for ns, fts in ns_desc.items():
        for f in fts.split(' '):
            for fi in _find_matches(f, features, keys=lambda i: ([i[0], i[1][0]]) ):
                ns_tree[ns].append( fi[0] )
                features.remove( fi )

    try:
        for f in features:
            ns_tree[ ns_chars.pop(0) ].append( f[0] )
    except IndexError as e:
        _exception("Too many features!\n     Alphabet is too short to be used as namespaces.\n     Please declare some namespaces on your config")

    return ns_tree 

def _columns_iterator(header, config, with_aliases=False, order='alphabetical', memoize=False):
    if with_aliases:
        for a,d in config['aliases'].items():
            if d:
                tokens  = d.split(' ')
                cb, sep = transformer(tokens[0], ' '.join(tokens[1:]) or 'raw', memoize=memoize)
                yield a, (tokens[0], cb, "'" if a == 'tag' else None)

    features = {}
    for k,defs in sorted(config['features'].items()):
        cols = _find_matches(k, header)
        defs = defs if type(defs) == list else [ defs ]
        for h in cols or [k]:
            for i,_def in enumerate(defs):
                name    = h + str(i if len(defs) > 1 else '')
                cb, sep = transformer(name, _def, memoize=memoize)
                order_v = ( len(cols) == 0, header.index(h) if order == 'header' and len(cols) else name )
                features[name] = ( order_v, name, (h,cb,sep) )

    for _,n,x in sorted(features.values()):
        yield n,x

def _find_matches(needle, items, keys=lambda i: [i]):
    search  = needle.replace("*", ".*") + '$'
    items   = sorted(items, key=lambda x: '*' not in x)
    match   = lambda i: sum([ bool(re.match(search, j)) for j in keys(i) ])
    return [ i for i in items if match(i) ]

def _possible_chars(ns_desc, limit=140):
    char_order = lambda x: \
        sum([ bool(re.match(r,x))*100 for r in ['[^a-z]', '[^a-zA-Z]', '[^a-zA-Z0-9]'] ]) + ord(x)
    allowed = lambda i:  '\\' not in repr(i) and not re.match(prohibited_chars, i) and i not in ns_desc
    return sorted([ i for i in map(chr,range(limit)) if allowed(i) ], key=char_order)

# input generation

def _dummy_parser(row):
    yield row

def _vw_generator(stream, config, parser=None, exps=[], header=[], indexes=False, memoize=True):
    parser = parser or _dummy_parser

    feature_list, vw_template = \
        get_feature_config(config, exps, indexes=indexes, header=header, memoize=memoize)

    for line in stream:
        for row in parser(line):
            features = [ t(row[i]) for i,t in feature_list ]
            yield vw_template % tuple(features)

# csv

def csv_to_vw(config, input='/dev/stdin', parser=None, sep="\t", exps=[], sample_rate=None ):
    stream    = open(input)
    header    = _get_csv_header(config, stream, sep=sep)

    my_parser = _csv_parser(parser, header=header, probability=sample_rate, sep=sep)
    indexes   = not bool(parser)

    for line in _vw_generator(stream, config, parser=my_parser, header=header, indexes=indexes, exps=exps):
        print(line)

def _get_csv_header(config, stream, sep="\t"):
    if 'header' not in config:
        return stream.readline().strip("\n").split(sep)
    return config['header'].split(",")

def _csv_parser(parser, header=None, probability=None, sep=None):
    header_str = sep.join(header)+"\n"

    def row_parser(line):
        if line != header_str and (probability is None or random.random() < probability):
            yield line.strip("\n").split(sep)

    def dict_parser(stream):
        for row in row_parser(stream):
            for row_dict in parser( dict(zip(header, row)) ):
                yield row_dict

    return dict_parser if parser else row_parser


# spark

def table_to_vw( table, config, parser=None, exps=[] ):
    df = __my_spark().table( table )
    rdd_to_vw(df.rdd, config, parser=parser, exps=exps, save_to=table + '_vw' )

def rdd_to_vw( rdd, config, parser=None, exps=[], save_partitions=100, save_to=None ):
    sc = __my_spark().sparkContext
    sc.addPyFile( os.path.abspath(__file__) )
    sc.addPyFile( os.path.dirname(os.path.abspath(__file__)) + "/vw_features.py" )

    dframe = rdd \
        .mapPartitions(lambda x: _vw_generator(x, config, parser=parser, exps=exps, memoize=False)) \
        .map(lambda x: [x]) \
        .toDF( [ 'lines' ] )

    if save_to:
        from pyspark.sql import functions as F
        dframe.repartition( save_partitions, F.rand() ) \
            .sortWithinPartitions( F.rand() ) \
            .write.csv(save_to, mode="overwrite")

    return dframe

def __my_spark():
    from pyspark.sql import SparkSession
    return SparkSession.builder.enableHiveSupport().getOrCreate()


# final api

def to_vw(config, input=None, spark=False, parser=None, sep="\t", exps=[], sample_rate=None, debug=False ):
    if spark:
        table_to_vw(input, config, parser=parser, exps=[] )
    else:
        csv_to_vw(config, input=input or '/dev/stdin', parser=parser, sep=sep, sample_rate=sample_rate, exps=exps )
