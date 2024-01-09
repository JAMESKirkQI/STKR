import torch
from torch.utils.data import TensorDataset, Subset
from torch.utils.data import DataLoader


class MultiviewDataset(TensorDataset):
    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors[:-1]), self.tensors[-1][index].long()


class FeatureLoader(object):
    def __init__(self, f_path='xy_tensor_uiuc.pt'):
        xy = torch.load(f_path)
        self.view1 = xy[0]
        self.view2 = xy[1]
        self.view3 = xy[2]
        self.view4 = xy[3]
        self.label = xy[4]
        self.num_labeled_data = 0
        self.num_unlabeled_data = 0

    def distinguish_dataloader(self, num_per_cls=70, cls=8):
        labeled_index, unlabeled_index = [], []
        for c in range(cls):
            labeled_index.append((self.label == c).nonzero(as_tuple=False).squeeze()[:num_per_cls])
            unlabeled_index.append((self.label == c).nonzero(as_tuple=False).squeeze()[num_per_cls:])

        labeled_indices = torch.concat(labeled_index)
        unlabeled_indices = torch.concat(unlabeled_index)
        dataset = MultiviewDataset(self.view1, self.view2, self.view3, self.view4, self.label)

        labeled_dataset = Subset(dataset, labeled_indices)
        self.num_labeled_data = len(labeled_dataset)

        unlabeled_dataset = Subset(dataset, unlabeled_indices)
        self.num_unlabeled_data = len(unlabeled_dataset)

        labeled_loader = DataLoader(labeled_dataset, batch_size=self.num_labeled_data, shuffle=False)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=64, shuffle=True)

        return labeled_loader, unlabeled_loader
