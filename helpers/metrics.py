from __future__ import print_function
from contextlib import contextmanager

import time as time
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import scipy.stats as st
import sklearn.metrics        as sk_metrics
import sklearn.calibration    as sk_cal

import warnings
warnings.filterwarnings("ignore")


def visualize_performance(true, pred, weights=None, bins=None, sample_size=None, layout=22, model='', tile=None, save_dir=None):
    true, pred, weights = np.array(true), np.array(pred), np.array(weights or np.ones(len(true)))

    # detecting type
    _type =  '_regression'
    if len(np.unique(true[:1000])) == 2:
        _type = '_classifier'
    if type(pred[0]) == list:
        _type = '_multiclass'
   
    # removing outliers
    if tile and _type == 'regression': 
        hist, edges = np.histogram(pred, bins=200, weights=weights)
        cum  = hist.cumsum() / hist.sum()
        i    = true < edges[ np.where(cum >= tile)[0][0] ]
        if i.sum() > 0: true, pred, weights = np.array(true)[i], np.array(pred)[i], np.array(weights)[i]
    
    # sampling
    if _type == 'regression':
        sample_size = sample_size or min(len(true), 30000)

    if sample_size:
        i = np.random.choice(list(range(len(true))), min(sample_size, len(true)), replace=False)
        true, pred, weights = np.array(true)[i], np.array(pred)[i], np.array(weights)[i]

    # plotting args
    bins  = bins or np.min((200, int(1+len(true)/70)))
    args  = { 'name': model, 'bins': bins, 'layout': layout }
    figsize = ( 10 * (layout % 10), 7 * (layout / 10) )
   
    # plotting
    os.system('mkdir -p '+save_dir)
    fig = plt.figure(figsize=figsize) # 20, 15
    fig.suptitle("Performance Plot ({} / {})".format(sample_size or len(true), len(true)))
    globals()[_type](true, pred, weights, args) 
    plt.savefig( save_dir + "/{}_perf.{}.png".format(model, time.strftime("%Y%m%d-%H%M%S")) )
    plt.close()


def _classifier(true, pred, weights, args):
    _plot_pr_curve            ( true, pred, weights, ax=_subplot(args,1), **args )
    _plot_decision_thresholds ( true, pred, weights, ax=_subplot(args,2), **args )
    _plot_classes_pdf         ( true, pred, weights, ax=_subplot(args,3), **args )
    _plot_calibration_curve   ( true, pred, weights, ax=_subplot(args,4), **args )

def _regression(true, pred, weights, args):
    _plot_pdf             ( [ true, pred ],  weights, ax=_subplot(args,1), bins=args['bins'],   label=[ 'Truth', 'Predicted' ] )
    _plot_cdf             ( [ true, pred  ], weights, ax=_subplot(args,2), bins=args['bins']*4, label=[ 'Truth', 'Predicted' ] )
    _plot_scatter         ( true, pred,      [np.min(true), np.max(true)], ax=_subplot(args,3), xlabel='Truth',     ylabel='Predicted', title='Response Plot')
    _plot_scatter         ( pred, pred-true, [0,0],                        ax=_subplot(args,4), xlabel='Predicted', ylabel='Residuals', title='Residual Plot')


# regression plotting

def _plot_scatter_heatmap(true, pred, ax=None, bins=None):
    heatmap, xedges, yedges = np.histogram2d(true, pred, bins=bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    heatmap = np.log(heatmap+1)
    a = list(range(int(np.min(true)), int(np.max(true))))
    
    ax.imshow(heatmap.T, extent=extent, origin='lower', interpolation='gaussian')
    ax.plot(a,a)

def _plot_scatter(x, y, ideal_y, ax=None, **kwargs):
    ax.scatter(x, y, marker='.')
    ax.plot([np.min(x), np.max(x)], ideal_y, 'k-')
    __meta( ax, **kwargs )

def _plot_pdf(data, weights=None, final_weights=None, ax=None, bins=None, label=None):
    if final_weights is None:
        final_weights = [ weights for i in data ]
    ax.hist(data, normed=True, bins=bins, label=label, weights=final_weights)
    __meta( ax, title='Response Histogram', xlabel="Predicted Values",  legend=True)

def _plot_cdf(data, weights, ax=None, bins=None, label=None):
    ax.hist(data, normed=True, cumulative=True, histtype='step', lw=2, bins=bins, label=label, weights=[ weights for i in data ])
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid()
    __meta( ax, title='Cumulative Distribution', xlabel="Predicted Values", ylim=[0,1.0],  legend=True)

# classification plotting

def _plot_classes_pdf(y_true, y_pred, weights, ax=None, **kwargs):
    classes = np.unique(y_true)
    data    = [ y_pred[y_true == v] for v in classes ]
    weights = [ weights[y_true == v] for v in classes ] 
    _plot_pdf( data, final_weights=weights, ax=ax, label=classes, bins=kwargs['bins'] )

def _plot_classes_cdf(y_true, y_pred, ax=None, **kwargs):
    bins, name = kwargs['bins'], kwargs['name']
    x = np.linspace( 0, 1, bins )
    total = np.array( bins * [ .0 ] )

    for v in np.unique(y_true):
        hist, _ = np.histogram(y_pred, bins, weights=(y_true == v) * 1.0)
        ax.plot(x, hist.cumsum() / hist.sum(), label="%s %d" % (name, v))
        total  += hist
    ax.plot(x, total.cumsum() / total.sum(), label="%s All" % name)

    __meta( ax, title='Decision Coverage', xlabel='Predicted Probability', ylabel='% of observations', ylim=[0,1], legend='lower right')

def _plot_calibration_curve(y_true, y_pred, weight, ax=None, **kwargs):
    y_pred = np.clip(y_pred, 1e-4, 1-1e-4)
    positives, mean = sk_cal.calibration_curve(y_true, y_pred, n_bins=50)
    
    brier = sk_metrics.brier_score_loss(y_true, y_pred, weight)
    ax.plot( mean, positives, label="Model %s | Brier score: %.3f" % (kwargs['name'], brier ))
    ax.plot( [0, 1], [0, 1], "k:", label="Perfectly calibrated" )

    __meta( ax, title='Calibration Curve', ylabel="% of positives", xlabel='Predicted Probability', xlim=[0,1], ylim=[0, 1], legend='lower right')


def _plot_roc_curve(y_true, y_pred, ax=None, **kwargs):
    fpr, tpr, thr = roc_curve(y_true, y_pred)
    auc = sk_metrics.roc_auc_score(y_true, y_pred)
    
    ax.plot(fpr, tpr, label="%s (auc %.3f)" % (kwargs['name'], auc ))
    __meta( ax, title='ROC Curve', ylabel='True Positive Rate', xlabel='False Positive Rate', legend='lower right', ylim=[0,1], xlim=[0,1] )

def _plot_pr_curve(y_true, y_pred, weights, ax=None, **kwargs):
    precision, recall, thresholds = sk_metrics.precision_recall_curve( y_true, y_pred, sample_weight=weights )
    lloss = sk_metrics.log_loss( y_true, y_pred, sample_weight=weights)
    auc   = sk_metrics.roc_auc_score(y_true, y_pred, sample_weight=weights)
    ax.plot(recall, precision, label="%s (LogLoss %.3f | AUC %.3f)" % (kwargs['name'], lloss, auc ) ) 
    
    base_rate = sum(y_true == 1) / float(len(y_true))
    __meta( ax, title='Precision Recall Curve', ylabel='Precision', xlabel='Recall', legend='lower left', ylim=[base_rate,1], xlim=[0,1])

def _plot_decision_thresholds(y_true, y_pred, weights, ax=None, **kwargs):
    precision, recall, pr_thr = sk_metrics.precision_recall_curve(y_true, y_pred, sample_weight=weights)
    fpr, tpr, roc_thr = sk_metrics.roc_curve(y_true, y_pred, sample_weight=weights)
    pos, neg = len(y_true[y_true == 1]), len(y_true[y_true == 0])

    ax.plot(pr_thr, precision[1:], label="%s Precision" % kwargs['name'])
    ax.plot(pr_thr, recall[1:],    label="%s Recall (TPR)" % kwargs['name'])
    ax.plot(roc_thr, (neg*(1-fpr) + pos*tpr)/len(y_true), label="%s Accuracy" % kwargs['name'])
    ax.plot(roc_thr, fpr, label="%s FPR" % kwargs['name'])
    ax.plot(pr_thr, 2.0 * precision[1:] * recall[1:] / ( precision[1:] + recall[1:] ), label="%s F1" % kwargs['name'])
    
    ax.set_xticks(np.arange(0, 1, 0.1))
    ax.set_xticks(np.arange(0, 1, 0.025), minor=True)
    ax.set_yticks(np.arange(0, 1, 0.1))
    ax.set_yticks(np.arange(0, 1, 0.025), minor=True)
    ax.grid()

    __meta( ax, title='Decision Thresholds', xlabel='Predicted Probability', legend='upper right', ylim=[0,1], xlim=[0,1] )

def __annotate_thresholds(x, y, thresholds, ax=None, n=10):
    n_points = 7
    for i in range(1, n_points):
        index = i * len(x) / (n_points + 1)
        ax.annotate("%.2f" % thresholds[index], xy=(x[index], y[index]), textcoords='data')

# helpers

def _subplot(args, _id):
    return plt.subplot(args['layout']*10 + _id)

def __meta( ax, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, legend=None ):
    ax_ = ax or plt
    title  and ax_.set_title(title)
    xlabel and ax_.set_xlabel(xlabel)
    ylabel and ax_.set_ylabel(ylabel)
    xlim   and ax_.set_xlim(xlim)
    ylim   and ax_.set_ylim(ylim)
    legend and ax_.legend(loc=legend)



