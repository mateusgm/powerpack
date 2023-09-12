from __future__ import absolute_import, print_function

import sys, datetime, time
import pickle
import os
import subprocess as sb

from contextlib import contextmanager


# subprocesses 

def run_and_stream_output(cmd, stdin=None):
    proc   = sb.Popen( cmd, shell=True, stdin=(stdin if stdin else sb.PIPE), stdout=sb.PIPE, stderr=sb.PIPE, universal_newlines=True )
    while proc.poll() is None:
        _, line = proc.stderr.readline(), proc.stdout.readline()
        if line.strip():
            yield line.strip()

# dates

def add_days(date, days):
    import datetime as dt
    _new = dt.datetime.strptime( date, "%Y-%m-%d" ) + dt.timedelta(days=days)
    return _new.strftime( "%Y-%m-%d" )  

def date_range(begin, days=None):
    current = begin
    while current < add_days(begin, days):
        yield current
        current = add_days(current, 1)

# logging 

@contextmanager
def suppress_warnings():
    import warnings
    old_warn = warnings.showwarning
    def warn(*args, **kwargs):
        pass
    
    warnings.showwarning = warn
    yield
    warnings.showwarning = old_warn

last_log = time.time()
def log(*args):
    global last_log
    print("%s [ %.1fm ]: %s" % ( time.strftime("%H:%M:%S"), (time.time() - last_log) / 60, " ".join([ str(a) for a in args ])), file=sys.stderr)
    last_log = time.time()


def picklefy(obj, name):
    if not os.path.exists('pkl'):
        os.makedirs('pkl')
    with open("pkl/%s.pkl" % (name), 'wb') as handle:
        pickle.dump(obj, handle)
