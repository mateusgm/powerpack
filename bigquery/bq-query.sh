#!/usr/bin/env python3.6
import subprocess as sb
import argparse
import re
import sys
import tempfile

bq_cmd  = 'bq query --nouse_legacy_sql'
cat_cmd = 'cat {} | '

parser = argparse.ArgumentParser()
parser.add_argument('--air',  nargs='*',            help='Airflow variables. Usage --air ds=2019-11-01 tomorrow_ds=2019-11-02')
parser.add_argument('--drop', action='store_true',  help='Drop table before creating it.')
parser.add_argument('--overwrite',                  help='Overwrite partition before inserting. Usage: --overwrite yyyymmdd,2019-10-01')
parser.add_argument('--debug', action='store_true', help='Print query and command instead of running it.')
parser.add_argument('--test_schema',                help="Save/insert to a test schema instead of the original one. Beware: it will replace all references to the table you're creating/inserting to.")
parser.add_argument('-f',    dest='file',           help='Query path')
args, bq_args = parser.parse_known_args()

if not args.file:
    _, args.file = tempfile.mkstemp()
    sb.call('cat > {}'.format(args.file), shell=True)

# getting values

full_sql = open(args.file).read()
prev     = 1
queries  = [''] + re.findall('(CREATE\s+TABLE.*|DELETE|INSERT|DROP)', full_sql)[1:][::-1]

while prev != 0:
    next = full_sql.index(queries.pop())
    sql  = full_sql[prev-1:next-1]
    prev = next

    # identifying the table

    src_table  = re.findall(r'CREATE\s+TABLE\s+([^\s]+)', sql)
    src_table += re.findall(r'CREATE\s+OR\s+REPLACE\s+TABLE\s+([^\s]+)', sql)
    src_table += re.findall(r'INSERT\s+INTO\s+([^\s]+)', sql)

    if src_table:
        table = src_table[0]

    if args.test_schema:
        tokens = table.split('.')
        table  = "{}.{}".format( args.test_schema, tokens[-1] )
        if len(tokens) == 3:
            table = "{}.{}.{}".format( tokens[0], args.test_schema, tokens[-1] )
        sql    = sql.replace(src_table[0], table)

    # composing the commands

    fd, file_path = tempfile.mkstemp()
    with open(fd, 'w') as f:
        f.write(sql)

    cat = cat_cmd.format( file_path )
    sed = ''
    if args.air:
        params = [ a.split('=') for a in args.air ]
        params += [ ( k+"_nodash", v.replace('-','')) for k,v in params ]
        sed    = ' | '.join([ "sed -e 's/{{ *%s *}}/%s/g'" % (k,v) for k,v in params ]) + ' | '

    cmd = '{} {} {} {}'.format(cat, sed, bq_cmd, ' '.join(bq_args))
    if args.debug:
        print(sql)
        print(cmd)
        continue


    # dropping and overwriting first

    if args.drop and table and 'dry_run' not in bq_args:
        sb.call( 'echo "DROP TABLE IF EXISTS {}" | {} {}'.format(table, sed, bq_cmd), shell=True )

    if args.overwrite and table and 'dry_run' not in bq_args:
        column,partition = args.overwrite.split(',')
        sb.call( "echo 'DELETE {} WHERE {}=\"{}\"' | {} {}".format(table, column, partition, sed, bq_cmd), shell=True )

    try:
        sb.check_call(cmd , shell=True )
    except sb.CalledProcessError:
        exit(1)
