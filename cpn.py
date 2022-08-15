import torch
import torch.nn as nn
import torch.nn.functional as F
class PrototypeClassifier(nn.Module):
    def __init__(self, dim_features, num_classes, centers=None):
        super().__init__()
        self.dim_features = dim_features
        self.num_calsses = num_classes
        if centers is not None:
            self.prototypes = nn.Parameter(torch.tensor(centers,dtype=float))
        else:
            self.prototypes = nn.Parameter(torch.randn(self.num_calsses, self.dim_features))

    def forward(self, x):
        x = x.reshape(-1, 1, self.dim_features)
        d = torch.pow(x - self.prototypes, 2)
        d = torch.sum(d, dim=2)
        return d

    def logits(self,x):
        d=self.forward(x)
        logits=F.softmax(-1.*d,dim=1)
        return logits
