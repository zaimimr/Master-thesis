
from ZeroCostFramework.utils.util_functions import initialise_zero_cost_proxy
from ZeroCostFramework.utils.zero_cost_proxy_interface import ZeroCostProxyInterface
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

def get_flattened_metric(net, metric):
    grad_list = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grad_list.append(metric(layer).flatten())
    flattened_grad = np.concatenate(grad_list)

    return flattened_grad


def get_grad_conflict(net, inputs, targets, loss_function=F.cross_entropy):
    N = inputs.shape[0]
    batch_grad = []
    for i in range(N):
        net.zero_grad()
        outputs = net.forward(inputs[[i]])
        if type(outputs) == tuple:
            outputs = outputs[0]
        loss = loss_function(outputs, targets[[i]])
        loss.backward()
        flattened_grad = get_flattened_metric(net, lambda
            l: l.weight.grad.data.cpu().numpy() if l.weight.grad is not None else torch.zeros_like(l.weight).cpu().numpy())
        batch_grad.append(flattened_grad)
    batch_grad = np.stack(batch_grad)
    direction_code = np.sign(batch_grad)
    direction_code = abs(direction_code.sum(axis=0))
    score = np.nanmean(direction_code)
    return score


class grad_sign(ZeroCostProxyInterface):
    def calculate_proxy(self, net, data_loader, device, loss_function, eval = False, train = True,  single_batch = True, bn = False) -> float:
        model, data, labels = initialise_zero_cost_proxy(net, data_loader, device, train=train, eval=eval,single_batch=single_batch, bn=bn)
        score = get_grad_conflict(model, data, labels, loss_function)
        del model
        model = None
        del data
        data = None
        del labels
        labels = None
        torch.cuda.empty_cache()
        return score