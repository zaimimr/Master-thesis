import numpy as np
import torch
from torch import nn
from ZeroCostFramework.utils.util_functions import get_layer_metric_array, initialise_zero_cost_proxy
from ZeroCostFramework.utils.zero_cost_proxy_interface import ZeroCostProxyInterface

class zen(ZeroCostProxyInterface):
    def calculate_proxy(self, net, data_loader, device, loss_function, eval = False, train = True, single_batch = True, bn = True, repeat=1, mixup_gamma=1e-2, fp16=False) -> float:
        model, data, labels = initialise_zero_cost_proxy(net, data_loader, device, train=train, eval=eval, single_batch=single_batch, bn=bn)
        nas_score_list = []

        dtype = torch.half if fp16 else torch.float32

        with torch.no_grad():
            for repeat_count in range(repeat):
                self.network_weight_gaussian_init(model)
                input = torch.randn(size=list(data.shape), device=device, dtype=dtype)
                input2 = torch.randn(size=list(data.shape), device=device, dtype=dtype)
                mixup_input = input + mixup_gamma * input2
                output = model.forward_before_global_avg_pool(input)
                mixup_output = model.forward_before_global_avg_pool(mixup_input)

                nas_score = torch.sum(torch.abs(output - mixup_output), dim=[1, 2, 3])
                nas_score = torch.mean(nas_score)

                # compute BN scaling
                log_bn_scaling_factor = 0.0
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        bn_scaling_factor = torch.sqrt(torch.mean(m.running_var))
                        log_bn_scaling_factor += torch.log(bn_scaling_factor)
                    pass
                pass
                nas_score = torch.log(nas_score) + log_bn_scaling_factor
                nas_score_list.append(float(nas_score))

        avg_nas_score = float(np.mean(nas_score_list))

        del model
        model = None
        del data
        data = None
        del labels
        labels = None
        torch.cuda.empty_cache()
        
        return avg_nas_score
    
    def network_weight_gaussian_init(self, net: nn.Module):
        with torch.no_grad():
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    if m.weight is None:
                        continue
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    continue
