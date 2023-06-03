import torch
from ZeroCostFramework.utils.util_functions import initialise_zero_cost_proxy
from ZeroCostFramework.utils.zero_cost_proxy_interface import ZeroCostProxyInterface


class params(ZeroCostProxyInterface):
    def calculate_proxy(self, net, data_loader, device, loss_function, eval = False, train = True, single_batch = True, bn = False) -> float:
        model, data, labels = initialise_zero_cost_proxy(net, data_loader, device, train=train, eval=eval, single_batch=single_batch, bn=bn)
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        del model
        model = None
        del data
        data = None
        del labels
        labels = None
        torch.cuda.empty_cache()
        return num_parameters