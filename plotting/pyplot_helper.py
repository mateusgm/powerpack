import matplotlib.pyplot as plt
import numpy as np

class PyPlotWrapper:
    calls   = []

    PLOTS   = [ 'scatter', 'plot', 'hist' ]
    METHODS = [ 'title' ]

    MAX_COLUMNS = 4
    COLUMN_INCH = 6

    def __init__(self):
        for m in np.append(self.METHODS, self.PLOTS):
            self.mirror_method(m)

    def mirror_method(self, method):
        def schedule_plot(*args, **kwargs):
            self.calls += [ { 'method': method, 'args': args, 'kwargs': kwargs } ]
            return self
        setattr(self, method, schedule_plot)

    def prepare(self, mask):
        plots = [ c for c in self.calls if c['method'] in self.PLOTS]
        n_plt = len(plots)

        mask = mask or (1 + n_plt/self.MAX_COLUMNS)*10 + ( self.MAX_COLUMNS if n_plt > self.MAX_COLUMNS else n_plt)
        size = ( self.COLUMN_INCH*(mask%10), self.COLUMN_INCH*int(mask / 10) )

        print("Printing %d charts" % (len(plots)))
        plt.figure(figsize=size)
        return mask*10 + 1

    def render(self, call, subplot):
        if call['method'] in self.PLOTS:
            plt.subplot(subplot)
            subplot += 1
        func = getattr(plt, call['method'])
        func(*call['args'], **call['kwargs'])
        return subplot

    def show(self, schema=None):
        try:
            subplot = self.prepare(schema)
            for p in range(len(self.calls)):
                subplot = self.render(self.calls[p], subplot)
        finally:
            plt.show()
            self.calls = []


