import argparse


def argparse_option():
    parser = argparse.ArgumentParser('Arguments for OPP-PersonReID')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=2, help='num of workers to use')
    # parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')

    # optimization
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')

    # model dataset
    parser.add_argument('--dataset', type=str, default='uiuc',
                        choices=['scene', 'uiuc', 'mit', 'cifar', 'caltech'], help='dataset')
    parser.add_argument('--iteration', type=int, default=150, help='optimization iteration')

    # other setting
    parser.add_argument('--lamb1', type=float, default=2.5, help='coefficient for the CVDC loss function')
    parser.add_argument('--lamb2', type=float, default=3.0, help='coefficient for the RCCA loss function')
    parser.add_argument('--sigma', type=float, default=1, help='parameter of Gaussian Kernel')
    parser.add_argument('--alpha', type=float, default=0.9, help='parameter of label propagation')
    parser.add_argument('--nearest', type=int, default=3, help='K nearest points')
    parser.add_argument('--seed', default=0, type=int, help='for reproducibility')

    opt = parser.parse_args()

    return opt
