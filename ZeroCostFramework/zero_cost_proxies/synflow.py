import torch
from ZeroCostFramework.utils.util_functions import initialise_zero_cost_proxy, get_score
from ZeroCostFramework.utils.zero_cost_proxy_interface import ZeroCostProxyInterface

class synflow(ZeroCostProxyInterface):
    def calculate_proxy(self, net, data_loader, device, loss_function, eval = False, train = True,  single_batch = True, bn = False) -> float:
        model, data, labels = initialise_zero_cost_proxy(net, data_loader, device, train=train, eval=eval,single_batch=single_batch, bn=bn)

        @torch.no_grad()
        def linearize(model):
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs


        @torch.no_grad()
        def nonlinearize(model, signs):
            for name, param in model.state_dict().items():
                if "weight_mask" not in name:
                    param.mul_(signs[name])

        signs = linearize(model)

        model.zero_grad()
        model.double()
        input_dim = list(data[0,:].shape)

        inputs = torch.ones([1] + input_dim).double().to(device)
        outputs = model.forward(inputs)
        if type(outputs) == tuple:
            outputs = outputs[0]
        torch.sum(outputs).backward()

        def synflow(layer):
            if layer.weight.grad is not None:
                return torch.abs(layer.weight * layer.weight.grad)
            else:
                return torch.zeros_like(layer.weight)

        nonlinearize(net, signs)
        score = get_score(model, synflow, "param")

        del model
        model = None
        del data
        data = None
        del labels
        labels = None
        torch.cuda.empty_cache()
        return score