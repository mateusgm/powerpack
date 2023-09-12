from __future__ import absolute_import, print_function

import sys

def consumer(callback, n=None, threads=False):
    import queue
    import multiprocessing as mp
    import threading as mt

    def worker(_input, _output):
        results = []
        for job in iter(_input.get, 'END'):
            results.append( callback(job) )
        _output.put( results )

    def wrapup(force=False):
        if not force:
            results = []
            for w in workers: _input.put('END')
            for w in workers: results.extend(_output.get())
            for w in workers: w.join()
            return results
        if not threads:
            for w in workers: w.terminate()

    conc    = n or mp.cpu_count()
    _input  = queue.Queue()
    _output = queue.Queue()

    workers = []
    for _ in range(n):
        w = mp.Process(target=worker, args=(_input,_output))
        if threads:
            w = mt.Thread(target=worker, args=(_input,_output))
        w.daemon = True
        w.start()
        workers.append(w)

    return _input.put, wrapup

def process_and_report(process, jobs, concurrency=10, threads=False):
    import multiprocessing.dummy as tp
    import multiprocessing as pp

    module  = tp if threads else pp
    pool    = module.Pool( concurrency )
    results = pool.imap_unordered( process, jobs )

    print("Starting {} jobs: {} ...".format( len(jobs), jobs[:3] ), file=sys.stderr)
    begin   = time.time()

    for i, r in enumerate(results, 1):
        print("%d / %d :: %.1f m/job" % ( i, len(jobs), float(time.time() - begin)/i/60 ), file=sys.stderr)
        yield r
