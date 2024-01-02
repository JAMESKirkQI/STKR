import torch
from torch.nn import MSELoss, KLDivLoss
import torch.nn.functional as F
import dgl


def omega(k):
    p = torch.arange(1, k + 1, 1)
    return torch.exp(-(p - 1) / k)


class Loss(object):
    def __init__(self, opt, device):
        self.mse = MSELoss()
        self.kl = KLDivLoss(reduction='batchmean', log_target=True)
        self.opt = opt
        self.device = device
        self.k = 0
        self.vskp_loss = []

    def caco(self, o):
        n = len(o)
        bsz = o[0].size(0)
        rwj = []
        caco_loss = 0
        for j in range(n):
            rwj.append(self.mse(o[j] @ o[j].t()/bsz, torch.eye(o[j].size(0)).to(self.device)))
            srio = []
            for i in range(n):
                if i == j:
                    continue
                else:
                    srio.append(torch.trace(o[j] @ o[i].t()) / (2*bsz))

            caco_loss += rwj[-1] - sum(srio)

        return caco_loss

    def cvdc(self, o):
        n = len(o)
        cvdc_loss = 0
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                else:
                    log_p = F.log_softmax(o[p])
                    log_q = F.log_softmax(o[q])
                    cvdc_loss += self.kl(log_p, log_q)

        return cvdc_loss

    def vskp(self, ol, ou, label, target, label_propagation, k, cls=8):
        n = len(ol)
        label = label.cpu()
        num_labeled_data = ol[0].size(0)
        num_unlabeled_data = ou[0].size(0)
        Yxl = torch.nn.functional.one_hot(label, num_classes=cls)
        Yxu = torch.nn.functional.one_hot(target, num_classes=cls)
        Y = torch.concat((Yxl, Yxu), dim=0).cpu()
        mask = torch.cat([torch.ones(num_labeled_data), torch.zeros(num_unlabeled_data)]).bool()
        vskp_loss = 0
        vote = []
        for v in range(n):
            Xv = torch.concat((ol[v], ou[v]), dim=0).clone().detach().cpu()
            knn_g = dgl.knn_graph(Xv, 3, exclude_self=True)
            new_labels = label_propagation(knn_g, Y, mask)

            # Yv = torch.zeros_like(Fv)
            # Yv[torch.arange(len(Fv)), Fv.argmax(1)] = 1
            Yv = new_labels[num_labeled_data:, :]
            vote.append(Yv)

        result = sum(vote)
        _, predicted = result.max(1)
        Cv = torch.zeros([num_labeled_data, num_unlabeled_data]).to(self.device)
        for i in range(num_unlabeled_data):
            indecies = (predicted[i] == label).nonzero().squeeze()
            Cv[indecies, i] = 1/len(indecies)

        for v in range(n):
            ouv = ou[v].clone().detach()
            vskp_loss += self.mse(ouv.t(), ol[v].t() @ Cv)
        vskp_loss /= n

        if self.k == k:
            self.vskp_loss.append(vskp_loss)
            self.k += 1
        w = omega(k+1)
        loss = 0
        for (i, lo) in enumerate(self.vskp_loss):
            loss += w[i] * lo
        loss /= (k+1)
        return loss
