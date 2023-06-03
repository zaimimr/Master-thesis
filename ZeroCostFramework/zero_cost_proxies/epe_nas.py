import numpy as np
import torch
from ZeroCostFramework.utils.util_functions import initialise_zero_cost_proxy
from ZeroCostFramework.utils.zero_cost_proxy_interface import ZeroCostProxyInterface 

class epe_nas(ZeroCostProxyInterface):
    def calculate_proxy(self, net, data_loader, device, loss_function, eval = False, train = True,  single_batch = True, bn = False) -> float:
    
        model, data, labels = initialise_zero_cost_proxy(net, data_loader, device, train=train, eval=eval, single_batch=single_batch, bn=bn)
        jacobs = []
        jac_labels = []
        
        try:
            jacobs_batch, target, n_classes = self.get_batch_jacobian(model, data, labels, None, None)
            jacobs.append(jacobs_batch.reshape(jacobs_batch.size(0), -1).cpu().numpy())

            if len(target.shape) == 2: # Hack to handle TNB101 classification tasks
                target = torch.argmax(target, dim=1)

            jac_labels.append(target.cpu().numpy())

            jacobs = np.concatenate(jacobs, axis=0)

            s = self.eval_score_perclass(jacobs, jac_labels, n_classes)
        except Exception as e:
            print(f"Error in calculating epe_nas: {e}")
            s = np.nan
        del model
        model = None
        del data
        data = None
        del labels
        labels = None
        torch.cuda.empty_cache()
        return s
    
    def get_batch_jacobian(self, net, x, target, to, device, args=None):
        net.zero_grad()
        x.requires_grad_(True)
        y = net(x)
        if type(y) == tuple:
            y = y[0]
        y.backward(torch.ones_like(y))
        jacob = x.grad.detach()
        return jacob, target.detach(), y.shape[-1]

    def eval_score_perclass(self, jacob, labels=None, n_classes=10):
        k = 1e-5
        per_class={}
        for i, label in enumerate(labels[0]):
            if label in per_class:
                per_class[label] = np.vstack((per_class[label],jacob[i]))
            else:
                per_class[label] = jacob[i]
        ind_corr_matrix_score = {}
        for c in per_class.keys():
            s = 0
            try:
                corrs = np.array(np.ma.corrcoef(per_class[c]))

                s = np.sum(np.log(abs(corrs)+k))#/len(corrs)
                if n_classes > 100:
                    s /= len(corrs)
            except: # defensive programming
                continue
            ind_corr_matrix_score[c] = s

        # per class-corr matrix A and B
        score = 0
        ind_corr_matrix_score_keys = ind_corr_matrix_score.keys()
        if n_classes <= 100:

            for c in ind_corr_matrix_score_keys:
                # B)
                score += np.absolute(ind_corr_matrix_score[c])
        else:
            for c in ind_corr_matrix_score_keys:
                # A)
                for cj in ind_corr_matrix_score_keys:
                    score += np.absolute(ind_corr_matrix_score[c]-ind_corr_matrix_score[cj])

            if len(ind_corr_matrix_score_keys) > 0:
                # should divide by number of classes seen
                score /= len(ind_corr_matrix_score_keys)

        return score
