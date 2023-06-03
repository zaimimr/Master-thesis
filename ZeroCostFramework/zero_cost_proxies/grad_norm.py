import torch
from ZeroCostFramework.utils.util_functions import initialise_zero_cost_proxy, get_score
from ZeroCostFramework.utils.zero_cost_proxy_interface import ZeroCostProxyInterface

class grad_norm(ZeroCostProxyInterface):
    def calculate_proxy(self, net, data_loader, device, loss_function, eval = False, train = True, single_batch = True, bn = True) -> float:
        model, data, labels = initialise_zero_cost_proxy(net, data_loader, device, train=train, eval=eval, single_batch=single_batch, bn=bn)
        split_data = 1
        
        while True:
            try:
                N = data.shape[0]
                for sp in range(split_data):
                    st = sp * N // split_data
                    en = (sp + 1) * N // split_data

                    outputs = model.forward(data[st:en])
                    if type(outputs) == tuple:
                        outputs = outputs[0]
                    loss = loss_function(outputs, labels[st:en])
                    loss.backward()

                    score = get_score(model, lambda l: l.weight.grad.norm() if l.weight.grad is not None else torch.zeros_like(l.weight), mode='param')
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
        

