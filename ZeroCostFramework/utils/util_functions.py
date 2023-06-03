import copy
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from datetime import date, datetime, timedelta
import types


def sum_arr(arr):
    sum = 0.
    for i in range(len(arr)):
        sum += torch.sum(arr[i])
    return sum.item()

def get_score(net, metric, mode):
    metric_array = get_layer_metric_array(net, metric, mode)
    return sum_arr(metric_array)

def no_op(self, x):
    return x

def initialise_zero_cost_proxy(net, data_loader, device, eval, train, single_batch, bn):
    model = copy.deepcopy(net)
    if bn == False:
        for l in model.modules():
            if isinstance(l, nn.BatchNorm2d) or isinstance(l, nn.BatchNorm1d):
                l.forward = types.MethodType(no_op, l)
    model.to(device)
    model.zero_grad()
    if train:
        model.train()
    if eval:
        model.eval()
        
    
    if single_batch:
        process = iter(data_loader)
        batch = next(process)
    
    data = Variable(batch[0].float().to(device), requires_grad=False) 
    labels = Variable(batch[1].long().to(device), requires_grad=False)

    torch.cuda.empty_cache()
    
    return model, data, labels

def reshape_elements(elements, shapes, device):
    def broadcast_val(elements, shapes):
        ret_grads = []
        for e,sh in zip(elements, shapes):
            ret_grads.append(torch.stack([torch.Tensor(sh).fill_(v) for v in e], dim=0).to(device))
        return ret_grads
    if type(elements[0]) == list:
        outer = []
        for e,sh in zip(elements, shapes):
            outer.append(broadcast_val(e,sh))
        return outer
    else:
        return broadcast_val(elements, shapes)
    
    
def calculate_function_runtime(function, *args, **kwargs):
    import time
    start_time = time.time()
    DEBUG = True
    if DEBUG:
        score = function(*args, **kwargs)
    else:
        try:
            score = function(*args, **kwargs)
        except Exception as e:
            print(e)
            score = None
    end_time = time.time()
    elapsed_time = end_time - start_time
    return (score, elapsed_time)

def get_layer_metric_array(net, metric, mode):
    metric_array = []

    for layer in net.modules():
        if mode == "channel" and hasattr(layer, "dont_ch_prune"):
            continue
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            metric_array.append(metric(layer))

    return metric_array


def get_proxies(exclude=[]):
    def files_filter(f):
        if f == '__init__.py':
            return False
        if f[:-3] in exclude:
            return False
        if f.endswith('.py'):
            return True
        return False
    proxies = [f.replace(".py", "") for f in os.listdir(f"{os.path.dirname(__file__)}/../zero_cost_proxies") if files_filter(f)]
    proxies.sort()
    return proxies

def np_json_parse(obj):
    if isinstance(obj, np.bool_):
            return bool(obj)
    if isinstance(obj, (np.floating, np.complexfloating)):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.string_):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, timedelta):
        return str(obj)
    return obj