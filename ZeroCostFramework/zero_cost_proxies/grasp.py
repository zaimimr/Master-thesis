import torch
from ZeroCostFramework.utils.util_functions import initialise_zero_cost_proxy, get_score
import torch.autograd as autograd

from ZeroCostFramework.utils.zero_cost_proxy_interface import ZeroCostProxyInterface

class grasp(ZeroCostProxyInterface):
    def grasp_func(self,layer):
        if layer.weight.grad is not None:
            return -layer.weight.data * layer.weight.grad   # -theta_q Hg
            #NOTE in the grasp code they take the *bottom* (1-p)% of values
            #but we take the *top* (1-p)%, therefore we remove the -ve sign
            #EDIT accuracy seems to be negatively correlated with this metric, so we add -ve sign here!
        else:
            return torch.zeros_like(layer.weight)
    def calculate_proxy(self, net, data_loader, device, loss_function, eval = False, train = True, single_batch = True, bn = True) -> float:
        # compute final sensitivity metric and put in grads

        model, data, labels = initialise_zero_cost_proxy(net, data_loader, device, train=train, eval=eval, single_batch=single_batch, bn=bn)
        split_data, num_iters, T = 1, 1, 1
        
        while True:
            try:
                # get all applicable weights
                weights = []
                for layer in model.modules():
                    if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                        layer.weight.requires_grad_(True)
                        weights.append(layer.weight)
                N = data.shape[0]
                for sp in range(split_data):
                    st = sp * N // split_data
                    en = (sp + 1) * N // split_data

                    # forward/grad pass #1
                    grad_w = None
                    for _ in range(num_iters):
                        # TODO get new data, otherwise num_iters is useless!
                        outputs = model.forward(data[st:en])
                        if type(outputs) == tuple:
                            outputs = outputs[0]
                        outputs = outputs / T
                        loss = loss_function(outputs, labels[st:en])
                        grad_w_p = autograd.grad(loss, weights, allow_unused=True)
                        if grad_w is None:
                            grad_w = list(grad_w_p)
                        else:
                            for idx in range(len(grad_w)):
                                grad_w[idx] += grad_w_p[idx]
                for sp in range(split_data):
                    st = sp * N // split_data
                    en = (sp + 1) * N // split_data

                    # forward/grad pass #2
                    outputs = model.forward(data[st:en]) 
                    if type(outputs) == tuple:
                        outputs = outputs[0]
                    outputs = outputs / T
                    loss = loss_function(outputs, labels[st:en])
                    grad_f = autograd.grad(loss, weights, create_graph=True, allow_unused=True)

                    # accumulate gradients computed in previous step and call backwards
                    z, count = 0, 0
                    for layer in model.modules():
                        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                            if grad_w[count] is not None:
                                z += (grad_w[count].data * grad_f[count]).sum()
                            count += 1
                    z.backward()
                
                score = get_score(model, self.grasp_func, "param")
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
