#!/usr/bin/env python3
import python.vowpal_helper as vw
import sys

config = {
    'aliases': {
        'label':  '{label}',
        'tag':    '{tag}',
        'weight': '{weight}',
    },
    
    'features': {
        # types: bool, cat, list, int, raw, num <func>, bin <limits>, clip <min max>
        {features}
    },

    'header': '{header}',

    # 'namespaces': {
        # 'a': "feature_regex* feature1",
    # }
}

def my_parser(row_dict):
    # do something and yield the row_dict (as many times as you want)
    yield row_dict

vw.csv_to_vw( config, sep="\t", parser=my_parser, exps=sys.argv[1:] )

