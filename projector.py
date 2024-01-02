from collections import OrderedDict

from torch import nn


class Projector(nn.Module):
    def __init__(self, input_dim=2048, output_dim=128, num_views=4, drop=0.0, cls=8):
        super(Projector, self).__init__()
        self.drop = drop
        self.num_views = num_views
        self.projectors = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        for i in range(num_views):
            if drop > 0:
                self.projectors.append(nn.Sequential(OrderedDict(
                            [('proj', nn.Linear(input_dim, output_dim, bias=False)), ('drop1', nn.Dropout(p=drop))]
                        )))
                self.classifiers.append(nn.Sequential(OrderedDict(
                    [('class', nn.Linear(output_dim, cls, bias=False)), ('drop2', nn.Dropout(p=drop))]
                )))
            else:
                self.projectors.append(nn.Sequential(OrderedDict(
                    [('proj', nn.Linear(input_dim, output_dim, bias=False))]
                )))
                self.classifiers.append(nn.Sequential(OrderedDict(
                    [('class', nn.Linear(output_dim, cls, bias=False))]
                )))

    def forward(self, x):
        output = [self.projectors[i](x[i]) for i in range(self.num_views)]
        logit = [self.classifiers[i](output[i]) for i in range(self.num_views)]
        return output, logit
