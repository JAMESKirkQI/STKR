import numpy as np
import torch
import random
from argpaser import argparse_option
from featureloader import FeatureLoader
from learner import Learner
from projector import Projector


# ssh -fgN -L 16006:127.0.0.1:6006 honestws@192.168.0.101
if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    opt = argparse_option()
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    loader = FeatureLoader()
    labeled_loader, unlabeled_loader = loader.distinguish_dataloader()
    projector = Projector().to(device)
    machine = Learner(unlabeled_loader, labeled_loader, projector, device, opt)
    machine.train()
