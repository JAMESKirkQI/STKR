import torch


def caculate_reliability(X, opt):
    A, B = [], []
    for x in X:
        x = x.clone().detach()
        bsz = x.size(0)
        dim = x.size(1)
        var = torch.var(x)

        a = torch.exp(-1*torch.cdist(x, x)/(dim*var))
        a.fill_diagonal_(0)
        A.append(a)
        _, idx = torch.topk(a, opt.nearest, dim=1)

        b = torch.zeros_like(a)
        for i in range(bsz):
            b[i, idx[i, :]] = 1
        B.append(b)

    sum_b = torch.zeros_like(b)
    for b in B:
        sum_b += b

    tz = (sum_b > 0).float()
    v = torch.concat([torch.reshape(a * tz, (-1,)).view(1, -1) for a in A], dim=0)
    r = torch.exp(-1 * torch.cdist(v, v)).fill_diagonal_(0)
    s = torch.sum(r, dim=1, keepdim=True).repeat(1, r.size(0))
    rio = r / s
    c = torch.sum(rio, dim=0) / torch.sum(rio)
    return rio, c
