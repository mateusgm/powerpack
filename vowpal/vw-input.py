#!/usr/bin/env python3
# vim: syntax=python
# 
# USAGE:
# 1. Generate your config file:    vw-input.py -init my_parser.py|config.vw.yml
# 2. Process your data:            cat dataset.csv | vw-input.py experiment1 experiment2
# 3. Be happy
#
from __future__ import print_function

import sys
import yaml
import os
import errno
import argparse
import python.vowpal_helper as vw

parser = argparse.ArgumentParser(description='')
parser.add_argument('exps',     help='Experiments that will overwrite default feature specifications',          nargs='*',                          default=[])
parser.add_argument('-c',       help='Configuration file, default: config.vw.yml',                              required=False, dest='config_file', default='config.vw.yml')
parser.add_argument('-i',       help='Input file. Default is stdin',                                            required=False, dest='input',       default='/dev/stdin')
parser.add_argument('-s',       help='Separator for input file columns',                                        required=False, dest='sep',         default="\t")
parser.add_argument('-p',       help='Sampling probability',                                                    required=False, dest='sample',      default=None, type=float)
parser.add_argument('--init',   help='Bootstrap a new parser config',                                           required=False, dest='init',        default='')
parser.add_argument('--debug',  help='Debug mode: print feature transforms' ,                                   required=False, dest='DEBUG',       default=False, action='store_true')
args = parser.parse_args()


# reading/changing config

def read_config(config_file):
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.scanner.ScannerError as e:
        print("Error when parsing the yaml:\n" + str(e), file=sys.stderr)
        exit(1)
    return config 

def init_python(path, template):
    repo_dir  = os.path.dirname(os.path.realpath(__file__))
    code_tmpl = open("{}/examples/my_vw_parser.py".format(repo_dir), 'r').read()
    features  = [ "'%s': '%s'" % x for x in sorted(template['features'].items()) ]
    with open(path, 'w') as f:
        code_tmpl = code_tmpl \
            .replace('{features}', ",\n        ".join(features)) \
            .replace('{header}', template['header']) \
            .replace('{label}',  template['aliases']['label']) \
            .replace('{tag}',    template['aliases']['tag']) \
            .replace('{weight}', template['aliases']['weight'])
        f.write(code_tmpl)
    os.system('chmod 755 {}'.format( path ) )

def init_yaml(config_file, template):
    yml = yaml.dump(template, default_flow_style=False) \
        .replace('features:', "features:\n  # types: bool, cat, list, int, raw, num <func>, bin <limits>, clip <min max>") \
        .replace('namespaces: {}', '#namespaces:\n#  a: "feature1 feature2 feature_regex*"')
    with open(config_file, 'w') as f:
        f.write(yml)


if args.init:
    print(">> Generating {}".format(args.init), file=sys.stderr)
    
    template = dict(
        aliases    = dict( label='col3' or '', tag='', weight=''),
        namespaces = dict(),
        header     =  ",".join(('col1','col2','col3')),
        features   = dict( col1='cat', col2= 'num'),
    )

    if '.py' in args.init:
        init_python( args.init, template )

    else:
        init_yaml( args.config_file, template )

    exit()


# generate vw input

config = read_config(args.config_file)

try:
    vw.to_vw(config, \
        input=args.input, \
        debug=args.DEBUG, \
        sep=args.sep, \
        exps=args.exps, \
        sample_rate=args.sample )

except KeyboardInterrupt:
    exit()

except Exception as e:
    if hasattr(e, 'errno') and e.errno == errno.EPIPE:
        exit()
    raise

