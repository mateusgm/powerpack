from __future__ import absolute_import, print_function

import h2o
import logging

# import sys; sys.path.append( '/home/mmelo/git_tree/mmelo-misc/helpers')
# import h2o_helper as mm_h
# import h2o
# from h2o.estimators.gbm import *
# from h2o.estimators.glm import *
# from h2o.estimators.random_forest import *
# from h2o.estimators.kmeans import *
# h2o_ctx = mm_h.init_h2o(spark, nodes=50, memory="10G")

# model

def sorted_grid(grid, metric='auc', decreasing=True):
    models = grid.get_grid(sort_by=metric, decreasing=decreasing)
    best   = h2o.get_model(models['Model Id'][0])
    return best, models

def save_coefs(model, name):
    coefs = pd.DataFrame(model.coef().items(), columns=['feature', 'coef'])
    coefs['coef_abs'] = np.abs(coefs['coef'])
    hlp.picklefy(coefs, "coefs_%s" % (name))
    return coefs

def export(model, name=None, path=None):
    path = path or ("/home/mmelo/exports/" + name)
    if not os.path.exists(path): os.makedirs(path)
    model.model_id = name
    model.download_pojo(path=path)

def evaluate_model(model, test, k=10000, label=None):
    import performance_helper as perf

    print("AUC: ",      model.auc(), model.auc(valid=True))
    print("LOSS: ",     model.logloss(), model.logloss(valid=True))
    print("DETAILS: ",  model.summary())
    print("VAR IMP:\n", model.varimp(use_pandas=True)[['variable', 'scaled_importance']])
    
    sample, _ = test_original.split_frame([ float(k) / test_original.dim[0] ])
    y_pred = model.predict(sample)[:,1].as_data_frame()
    y_true = sample[:,label].as_data_frame()
    print(y_pred[:10], y_true[:10])
    perf.visualize_performance( y_true, y_pred ) 


# setup

def init_h2o(spark, **kwargs):
    ctx = init_cluster(spark, **kwargs)
    get_flow_ssh(ctx)
    return ctx

def init_cluster(spark, nodes=10, memory="10G"):
    from pysparkling import *
    conf = H2OConf(spark) \
        .use_auto_cluster_start() \
        .set_yarn_queue("h2o") \
        .set_num_of_external_h2o_nodes(nodes) \
        .set_mapper_xmx(memory) \
        .set_cluster_start_timeout(1000)
    return H2OContext.getOrCreate(spark, conf)

def get_flow_ssh(context):
    import re, socket
    print("FLOW SSH: ssh -L %s:%s:%s %s -N" % (
          str(context._client_port)
        , str(context._client_ip)
        , str(context._client_port)
        , socket.gethostname()
    ))
    print("FLOW URL: http://127.0.0.1:" + str(context._client_port))

