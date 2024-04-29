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
    parser.add_argument('--diffusion_iteration', type=int, default=100, help='optimization diffusion_iteration')

    # other setting
    parser.add_argument('--RECT', action='store_true', default=True, help='RECT loss function')
    parser.add_argument('--lambRECT', type=float, default=2, help='coefficient for the RECT loss function')
    parser.add_argument('--CACO', action='store_true', default=False, help='CACO loss function')
    parser.add_argument('--lambCACO', type=float, default=2.5, help='coefficient for the CACO loss function')
    parser.add_argument('--CVDC', action='store_true', default=False, help='CVDC loss function')
    parser.add_argument('--lambCVDC', type=float, default=2, help='coefficient for the CVDC loss function')
    parser.add_argument('--CPRR', action='store_true', default=True, help='CPRR loss function')
    parser.add_argument('--lambCPRR', type=float, default=1, help='coefficient for the CPRR loss function')
    parser.add_argument('--upper_bound', type=float, default=500,
                        help='upper bound of the negative pairs distance,which is positive correlation with dimensions')
    parser.add_argument('--avg_select', action='store_true', default=True,
                        help='choose the method of sampling "True-->average sampling" "False-->random sampling" ')
    parser.add_argument('--sigma', type=float, default=1, help='parameter of Gaussian Kernel')
    parser.add_argument('--sigmaDG', type=float, default=1, help='parameter of Gaussian Kernel in diffusion')
    parser.add_argument('--alpha', type=float, default=0.9, help='parameter of label propagation')
    parser.add_argument('--alphaRECT', type=float, default=0.9, help='parameter of Graph diffusion')
    parser.add_argument('--nearest', type=int, default=3, help='K nearest points')
    parser.add_argument('--k_nums', type=int, default=0,
                        help='K nearest points in diffusion if k==0 then full connection')
    parser.add_argument('--scale', action='store_true', default=False, help='K nearest points')
    parser.add_argument('--CS', action='store_true', default=False)
    parser.add_argument('--CS_iterations', type=int, default=100, help='CS iteration')

    parser.add_argument('--seed', default=0, type=int, help='for reproducibility')

    opt = parser.parse_args()

    return opt
