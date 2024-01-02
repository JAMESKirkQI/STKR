import numpy as np
import torch
import dgl
from dgl.nn import LabelPropagation
from torch import nn
from torch.optim import Adam
from loss import Loss
from tensorboardX import SummaryWriter


class Learner(object):
    def __init__(self, unlabeled_dataloader, labeled_dataloader, projector, device, opt, cls=8):
        self.unlabeled_dataloader = unlabeled_dataloader
        self.labeled_dataloader = labeled_dataloader
        self.projector = projector
        self.device = device
        self.optimizer = Adam(projector.parameters(),
                             lr=opt.lr,
                             weight_decay=opt.weight_decay)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.opt = opt
        self.cls = cls
        self.mvm_loss = Loss(opt, device)
        num_tasks = len(self.unlabeled_dataloader)
        self.acc_array = np.zeros((num_tasks, num_tasks))
        self.label_propagation = LabelPropagation(k=400, alpha=0.9, clamp=False, normalize=True)
        self.writer = SummaryWriter("logs")

    def compute_avg_fgt(self):
        best_acc = np.max(self.acc_array, axis=1)
        end_acc = self.acc_array[-1]
        final_forgets = best_acc - end_acc
        avg_fgt = np.mean(final_forgets) * 100
        print(
            'The averaged forgetting of previous observed data: %.2f%%.'
            % avg_fgt
        )


    def test(self, k):
        self.projector.eval()
        labeled_data_iter = iter(self.labeled_dataloader)
        xl, label = labeled_data_iter.__next__()
        n = len(xl)
        xl = [xl[i].to(self.device) for i in range(n)]
        Yxl = torch.nn.functional.one_hot(label, num_classes=self.cls)
        num_labeled_data = xl[0].size(0)

        num_observed_data = 0
        num_correct_pred = 0

        for i, (xu, target) in enumerate(self.unlabeled_dataloader):
            if i > k:
                break

            Yxu = torch.nn.functional.one_hot(target, num_classes=self.cls)
            xu = [xu[i].to(self.device) for i in range(n)]
            num_unlabeled_data = xu[0].size(0)
            mask = torch.cat([torch.ones(num_labeled_data), torch.zeros(num_unlabeled_data)]).bool()
            ol, _ = self.projector(xl)
            ou, _ = self.projector(xu)

            Y = torch.concat((Yxl, Yxu), dim=0)

            vote = []
            for v in range(n):
                Xv = torch.concat((ol[v], ou[v]), dim=0).clone().detach().cpu()
                knn_g = dgl.knn_graph(Xv, 3, exclude_self=True)

                new_labels = self.label_propagation(knn_g, Y, mask)

                # Yv = torch.zeros_like(new_labels)
                # Yv[torch.arange(len(new_labels)), new_labels.argmax(1)] = 1
                Yv = new_labels[num_labeled_data:, :]
                vote.append(Yv)

            result = sum(vote)
            _, predicted = result.max(1)
            cur_num_correct_pred = predicted.eq(target).sum().item()
            self.acc_array[k, i] = cur_num_correct_pred / len(target)
            num_correct_pred += cur_num_correct_pred
            num_observed_data += len(target)

        acc = np.mean(self.acc_array[k,:k+1]) * 100
        print(
            'The current task number: %d, the averaged accuracy of previous observed data: %.2f%%.'
            % (k, acc)
        )
        self.projector.train()
        return acc

    def train(self):
        self.projector.train()
        labeled_data_iter = iter(self.labeled_dataloader)
        xl, label = labeled_data_iter.__next__()
        n = len(xl)
        xl = [xl[p].to(self.device) for p in range(n)]
        label = label.to(self.device)

        train_acc = []
        for i in range(n):
            for it in range(self.opt.iteration):
                _, logit = self.projector(xl)
                ce_loss = self.cross_entropy_loss(logit[i], label)
                self.optimizer.zero_grad()
                ce_loss.backward()
                self.optimizer.step()
            _, predicted = logit[i].max(1)
            total = len(label)
            correct = predicted.eq(label).sum().item()
            train_acc.append(correct / total)

        test_acc = []
        self.projector.eval()
        for i in range(n):
            total = 0
            correct = 0
            for k, (xu, target) in enumerate(self.unlabeled_dataloader):
                xu = [xu[p].to(self.device) for p in range(n)]
                target = target.to(self.device)
                _, logit = self.projector(xu)
                _, predicted = logit[i].max(1)
                total += len(target)
                correct += predicted.eq(target).sum().item()
            test_acc.append(correct / total)

        # self.test(10000)

        counter = 0

        for k, (xu, target) in enumerate(self.unlabeled_dataloader):
            xu = [xu[p].to(self.device) for p in range(n)]
            for i in range(self.opt.iteration//30):
                ol, _ = self.projector(xl)
                ou, _ = self.projector(xu)
                o = [torch.concat([ol[i], ou[i]], dim=0) for i in range(n)]
                caco_loss = self.mvm_loss.caco(o)
                cvdc_loss = self.mvm_loss.cvdc(o)
                semi_loss = self.mvm_loss.vskp(ol, ou, label, target, self.label_propagation, k)
                loss = semi_loss + self.opt.lamb1 * cvdc_loss - self.opt.lamb2 * caco_loss
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                self.writer.add_scalar("CACO loss", caco_loss.item(), global_step=counter)
                self.writer.add_scalar("CVDC loss", cvdc_loss.item(), global_step=counter)
                self.writer.add_scalar("Semi loss", semi_loss.item(), global_step=counter)
                counter += 1

            self.test(k)

        self.compute_avg_fgt()

