import random
import time

import torch
import wandb
from torch.nn import MSELoss, KLDivLoss
import torch.nn.functional as F
import dgl
import numpy as np


def omega(k):
    p = torch.arange(1, k + 1, 1)
    return torch.exp(-(p - 1) / k)


# define gaussian kernel function
def distance_matrix(X, scale=True, k=0, sigma=1, clamp=False, device="cuda:0"):
    if scale:
        sigmoid = torch.nn.Sigmoid()
        X = sigmoid(X)
    n = X.shape[0]
    G = X @ X.t()
    H = G.diag().repeat(n, 1)
    candidate = H + H.t() - G * 2
    mask = torch.ones_like(candidate, device=device) - torch.eye(n, device=device)
    E = torch.exp(-(candidate * mask) / 2.0 * sigma ** 2)
    edge = torch.exp(-candidate.detach() / 2.0 * sigma ** 2)
    if k:
        values, indices = edge.topk(k, dim=1)
        for i in range(indices.shape[0]):
            for j, indice in enumerate(indices[i]):
                edge[i, indice] = values[i, j]
        edge = (edge + edge.t()) / 2
    if clamp:
        edge = edge.clamp_(0.0, 1.0)
    assert torch.allclose(edge, edge.t(), rtol=1e-05, atol=1e-08)
    return edge, E


class Loss(object):
    def __init__(self, opt, device):
        self.softmax = torch.nn.Softmax(dim=-1)
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
            rwj.append(self.mse(o[j] @ o[j].t() / bsz, torch.eye(o[j].size(0)).to(self.device)))
            srio = []
            for i in range(n):
                if i == j:
                    continue
                else:
                    srio.append(torch.trace(o[j] @ o[i].t()) / (2 * bsz))

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

    def cprr(self, ol, ou, label, result, upper_bound_m=1, avg_select=True, device="cuda:0"):
        n = len(ol)
        cls = result.shape[-1]
        num_unlabeled_data = ou[0].size(0)
        sample = torch.zeros(cls, ol[0].size(-1), device=device)
        _, predicted = result.max(1)
        upper_bound_vector = torch.ones(cls, device=device) * upper_bound_m
        for i in range(len(sample)):
            index = torch.zeros(n, ol[0].size(-1))
            for j in range(n):
                if avg_select:
                    index[j] = ol[j][label == i].mean(dim=0)
                else:
                    selection = ol[j][label == i]
                    rand = random.randint(0, selection.shape[0] - 1)
                    index[j] = selection[rand]
            sample[i] = index.mean(dim=0)
        cprr_loss = 0
        for u in ou:
            for j in range(num_unlabeled_data):
                distance = ((u[j] - sample) @ (u[j] - sample).t()).diag()
                mask = torch.ge(upper_bound_vector, distance).int()
                mask[predicted[j]] = 1
                distance = upper_bound_m - distance
                distance[predicted[j]] = upper_bound_m - distance[predicted[j]]
                distance *= mask
                cprr_loss += torch.sum(distance)
        return cprr_loss / (n * num_unlabeled_data * cls)

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
            Cv[indecies, i] = 1 / len(indecies)

        for v in range(n):
            ouv = ou[v].clone().detach()
            vskp_loss += self.mse(ouv.t(), ol[v].t() @ Cv)
        vskp_loss /= n

        if self.k == k:
            self.vskp_loss.append(vskp_loss)
            self.k += 1
        w = omega(k + 1)
        loss = 0
        for (i, lo) in enumerate(self.vskp_loss):
            loss += w[i] * lo
        loss /= (k + 1)
        return loss, result

    def rect(self, ol, ou, k_nums, iterations, alpha, sigma, scale=True, CS=False, CS_iterations=400):
        n = len(ol)
        num_labeled_data = ol[0].size(0)
        A_all = []
        edges = []
        ES = []
        similarities = []
        rect_loss = torch.zeros(1, device=self.device)
        for v in range(n):
            Xv = torch.concat((ol[v], ou[v]), dim=0)
            bs = Xv.shape[0]
            # build the edge martix
            # edge = gaussian_kernel(Xv, k=k_nums, sigma=sigma, clamp=True, device=self.device)
            edge, E = distance_matrix(Xv, scale=True, k=k_nums, sigma=sigma, clamp=True, device=self.device)
            degree = (torch.ones(bs, device=self.device) / torch.sum(edge, dim=0).sqrt()).diag()
            similarity = degree @ edge @ degree
            ES.append(E)
            edges.append(edge)
            A_all.append(edge)
            similarities.append(similarity)
        A_all = torch.stack(A_all)
        similarities = torch.stack(similarities)
        edges = torch.stack(edges)
        ES = torch.stack(ES)
        A_tmp = torch.ones_like(A_all)
        for _ in range(iterations):
            # compute A
            for i in range(n):
                A_other = (torch.sum(A_all, dim=0) - A_all[i]) / (n - 1)
                A_tmp[i] = alpha * similarities[i] @ A_other @ similarities[i].t() + (1 - alpha) * edges[i]
                A_tmp[i] = A_tmp[i].clamp_(0.0, 1.0)
            if torch.allclose(A_all, A_tmp):
                break
            A_all = A_tmp.clone()
        A_avg = torch.mean(A_all, dim=0).detach()
        if CS:
            CS_alpha = 1 / num_labeled_data
            error_matrix = (torch.mean(edges, dim=0) - A_avg)
            error_matrix[num_labeled_data:, ] = 0
            similarity_avg = torch.mean(similarities, dim=0)
            error_t = error_matrix.clone()
            for _ in range(CS_iterations):
                error_tmp = (1 - CS_alpha) * error_matrix + CS_alpha * similarity_avg @ error_t @ similarity_avg.t()
                if torch.allclose(error_t, error_tmp):
                    break
                error_t = error_tmp
            err_sigma = torch.sum(error_matrix).div(num_labeled_data)
            A_avg += (error_t.t() * torch.div(err_sigma, torch.sum(error_t, dim=1))).t().detach()
        # compute the RECT loss
        for i in range(n):
            mask = torch.ones_like(edges[i], device=self.device) - torch.eye(edges[i].shape[0], device=self.device)
            bs = mask.shape[0]
            A_avg = mask * A_avg
            rect_loss += torch.sum((ES[i] - A_avg) ** 2) / (bs * (bs - 1))
        # observe the RDP loss behind the iterations
        # rdp_loss = torch.zeros(1, device=self.device)
        # for i in range(n):
        #     rdp_loss += torch.trace(
        #         A_all[i].t() @ A_all[i] - A_all[i].t() @ similarities[i].t() @ A_all[i] @ similarities[i]) \
        #                 + torch.sum((A_all[i] - edges[i]) ** 2)
        return rect_loss / n
