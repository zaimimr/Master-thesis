from ZeroCostFramework.utils.util_functions import get_score
from ZeroCostFramework.utils.zero_cost_proxy_interface import ZeroCostProxyInterface

class l2_norm(ZeroCostProxyInterface):
    def calculate_proxy(self, net, data_loader, device, loss_function, eval = False, train = True, single_batch = True, bn = False) -> float:
        return get_score(net, lambda l: l.weight.norm(), mode="param")