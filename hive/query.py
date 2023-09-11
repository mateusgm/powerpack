#!/usr/bin/spark-submit
from __future__ import print_function
import logging
import argparse
import sys
import os
import re
import sys
from datetime import datetime

from pyspark.sql import SparkSession

parser = argparse.ArgumentParser(description='This is a utility that will run standard hive queries in Spark')
parser.add_argument('-f','--file', help='supply the filepath where the hive script is stored')
parser.add_argument('-e','--execute', help='supply a string directly with the query you want to run', type=str)
parser.add_argument('--hivevar', help='pass a named variable to query: name=value', action='append')
parser.add_argument('--hiveconf', help='pass a named variable to query: name=value', action='append')
parser.add_argument('--outputformat', help='determines the separator of the output, possible values [csv,tsv]')
parser.add_argument('-v','--version', action='version', version='Spark query util v1.0')

args = vars(parser.parse_args())

fd = os.open('/dev/tty', os.O_WRONLY | os.O_NOCTTY)
tty = os.fdopen(fd, 'w', 1)
del fd

##################################
###### Defining functions #######
##################################

def print_with_time(printable_string):
    print(
        "[" +
        datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
        "]: " +
        str(printable_string)
    , file=tty
    )

def print_error_with_time(printable_string):
    print(
        "[" +
        datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
        "]: [ERROR]: " +
        str(printable_string)
    , file=sys.stderr
    )

def transform_to_python_string(input_field):
    if isinstance(input_field,unicode):
        return input_field.encode('utf-8')
    elif isinstance(input_field,basestring):
        return input_field
    else:
        return str(input_field)

def create_set_statements(hivevar_values, hiveconf_values):
    hivevar_pattern = "SET hivevar:{name}={value}"
    hiveconf_pattern = "SET {name}={value}"
    hv = [hivevar_pattern.format(name=n, value=v) for n, v in hivevar_values] 
    hc = [hiveconf_pattern.format(name=n, value=v) for n, v in hiveconf_values] 
    return hv + hc

def execute_query_and_print_results(query,spark_context):
    try:
        result = spark_context.sql(query)
    except Exception as e:
        print_with_time("Query couldn't run and ended with the following errors:")
        try:
            print_with_time(e.desc)
            print_error_with_time(e.desc)
        except:
            print_with_time(
                transform_to_python_string(e)
            )
            print_error_with_time(
                transform_to_python_string(e)
            )
        sys.exit(1)

    counter = 0
    for x in result.collect():
        if counter == 0:
            print(OUTPUTFORMAT.join(str(name) for name in x.__fields__))
        L = []
        for field in x:
            python_string = transform_to_python_string(field)
            if field is None:
                L.append("NULL")
            elif isinstance(field,unicode):
                L.append('"' + python_string + '"')
            elif isinstance(field,basestring):
                L.append('"' + python_string + '"')
            else:
                L.append(python_string)
        print(OUTPUTFORMAT.join(L))
        counter += 1

##################################
###### Parsing arguments   #######
##################################

if args['file']:
    print_with_time("Preparing to run file %s with spark" % args['file'])
    with open(args['file'], 'r') as infile:
        queries = " ".join(infile.readlines())
elif args['execute']:
    print_with_time("Preparing to run string %s with spark" % args['execute'])
    queries = args['execute']
else:
    print_with_time("ERROR - was not able to understand the input parameters, please look at the help function for usage")
    parser.print_help(sys.stderr)
    sys.exit(1)

hivevar_values = []
if args['hivevar']:
    for named_var in args['hivevar']:
        name, value = named_var.split('=', 1)
        print_with_time("Read hivevar variable at runtime: %s=%s" % (name, value))
        hivevar_values.append((name, value))

hiveconf_values = []
if args['hiveconf']:
    for named_var in args['hiveconf']:
        name, value = named_var.split('=', 1)
        print_with_time("Read hiveconf variable at runtime: %s=%s" % (name, value))
        hiveconf_values.append((name, value))

if args['outputformat']:
    print_with_time("Setting the outputformat to %s" % args['outputformat'])
    if re.search(r'csv',args['outputformat']):
        OUTPUTFORMAT = ","
    elif re.search(r'tsv',args['outputformat']):
        OUTPUTFORMAT = "\t"
    else:
        print_with_time("ERROR - was not able to understand the --outputformat parameter, please look at the help function for usage")
        parser.print_help(sys.stderr)
        sys.exit(1)
else:
    OUTPUTFORMAT = ","



##################################
### Initialising the cluster #####
##################################

print_with_time("Creating a spark session")

spark = (SparkSession.builder
         .appName("interactive_query_cli")
         .config("spark.dynamicAllocation.enabled","true")
         .config("spark.shuffle.consolidateFiles","true")
         .config("spark.shuffle.service.enabled","true")
         .enableHiveSupport()
         .getOrCreate()
         )

##################################
### Main part of the program   ###
##################################

query_list = []
if len(hivevar_values) + len(hiveconf_values) > 0:
    query_list.extend(create_set_statements(hivevar_values, hiveconf_values))

counter = 0
for query in re.split(r"([^\\];)",queries):
    if counter % 2 == 0:
        query_list.append(query)
    else:
        query_list[-1] = query_list[-1] + query.rstrip(";")
    counter += 1

for query in query_list:
    if len(query) > 1: #len(query) will be 1 if there is an empty line or if the last query ended with a semicolon
        print_with_time("Going to run the following query:\n%s" % query)
        execute_query_and_print_results(query,spark)

print_with_time("Query successfully run")
