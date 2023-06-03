import torch
import numpy as np
from thop import profile
from ZeroCostFramework.utils.util_functions import initialise_zero_cost_proxy
from ZeroCostFramework.utils.zero_cost_proxy_interface import ZeroCostProxyInterface 

class flops(ZeroCostProxyInterface):
    def calculate_proxy(self, net, data_loader, device, loss_function, eval = False, train = True,  single_batch = True, bn = False) -> float:
        
        model, data, _labels = initialise_zero_cost_proxy(net, data_loader, device, train=train, eval=eval, single_batch=single_batch, bn=bn)
        dummy_data = torch.from_numpy(np.zeros(data.shape)).float().to(device)
        
        # macs = profile_macs(model, dummy_data) // 2
        macs, _params = profile(model, inputs=(dummy_data, ))
        num_flops = int(macs * 2)
        del model
        model = None
        del data
        data = None
        del _labels
        _labels = None
        torch.cuda.empty_cache()
        return num_flops