import torch
import torch.nn as nn
import torch.nn.functional as F
from ZeroCostFramework.utils.util_functions import initialise_zero_cost_proxy, get_score

import types

from ZeroCostFramework.utils.zero_cost_proxy_interface import ZeroCostProxyInterface

class snip(ZeroCostProxyInterface):
    
    def snip_forward_conv2d(self, f_self, x):
            return F.conv2d(x, f_self.weight * f_self.weight_mask, f_self.bias,
                            f_self.stride, f_self.padding, f_self.dilation, f_self.groups)

    def snip_forward_linear(self, f_self, x):
            return F.linear(x, f_self.weight * f_self.weight_mask, f_self.bias)

    def snip_func(self, layer):
            if layer.weight_mask.grad is not None:
                return torch.abs(layer.weight_mask.grad)
            else:
                return torch.zeros_like(layer.weight)

    def calculate_proxy(self, net, data_loader, device, loss_function, eval = False, train = True, single_batch = True, bn = True) -> float:
        model, data, labels = initialise_zero_cost_proxy(net, data_loader, device, train=train, eval=eval, single_batch=single_batch, bn=bn)
        split_data, num_iters, T = 1, 1, 1
        
        while True:
            try:
                for layer in model.modules():
                    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                        layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
                        layer.weight.requires_grad = False
                    if isinstance(layer, nn.Conv2d):
                        layer.forward = types.MethodType(self.snip_forward_conv2d, layer)

                    if isinstance(layer, nn.Linear):
                        layer.forward = types.MethodType(self.snip_forward_linear, layer)

                N = data.shape[0]
                for sp in range(split_data):
                    st = sp * N // split_data
                    en = (sp + 1) * N // split_data

                    outputs= model.forward(data[st:en])
                    if type(outputs) == tuple:
                        outputs = outputs[0]
                    loss = loss_function(outputs, labels[st:en])
                    loss.backward()

                
                score = get_score(model, self.snip_func, "param")

                del model
                model = None
                del data
                data = None
                del labels
                labels = None
                torch.cuda.empty_cache()
                return score
            except RuntimeError as e:
                if "out of memory" in str(e):
                    done = False
                    if split_data == data.shape[0] // 2:
                        raise ValueError(
                            f"Can't split data anymore, but still unable to run. Something is wrong"
                        )
                    split_data += 1
                    while data.shape[0] % split_data != 0:
                        split_data += 1
                    torch.cuda.empty_cache()
                    print(f"Caught CUDA OOM, retrying with data split into {split_data} parts")
                else:
                    raise e