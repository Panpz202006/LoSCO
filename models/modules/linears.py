'''
Reference:
https://github.com/hshustc/CVPR19_Incremental_Learning/blob/master/cifar100-class-incremental/modified_linear.py
'''
import math

import torch
from torch import nn
from torch.nn import functional as F


class CosineLinear(nn.Module):
    """
    Classifier based on Cosine Similarity. 
    Each class can be represented by multiple(n>=1) learnable prototypes/proxies.

    Inputs:
    - in_features: feature dimension of input features
    - out_features: number of classes
    - nb_proxy: number of prototypes/proxies per class (default: 1)
    - to_reduce: whether to reduce multiple proxies into one
    - sigma: whether to use scaling parameter sigma
    """
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        
        # Learnable scaling parameter
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)
    
    def reset_parameters_to_zero(self):
        self.weight.data.fill_(0)

    def reduce_proxies(self, out, nb_proxy):
        if nb_proxy == 1:
            return out
        bs = out.shape[0]
        nb_classes = out.shape[1] / nb_proxy
        assert nb_classes.is_integer(), 'Shape error'
        nb_classes = int(nb_classes)

        simi_per_class = out.view(bs, nb_classes, nb_proxy)
        attentions = F.softmax(simi_per_class, dim=-1)

        return (attentions * simi_per_class).sum(-1)

    def forward(self, input):
        # Cosine similarity between normalized input and weights
        out = F.linear(
            F.normalize(input, p=2, dim=1), 
            F.normalize(self.weight, p=2, dim=1)
        )
        # Reduce proxy
        out = self.reduce_proxies(out, self.nb_proxy) if self.to_reduce else out
        # Scale with sigma
        out = self.sigma * out if self.sigma is not None else out

        return out
    
    def forward_task_agnostic(self, input, task_id, inc=10, feature_dim=768):
        # input shape [batch, 768]
        # task class slice
        start_cls = task_id * inc  
        end_cls = start_cls + inc
        
        weight_task = self.weight[start_cls:end_cls, :]

        # cosine similarity
        out = F.linear(
            F.normalize(input, p=2, dim=1), 
            F.normalize(weight_task, p=2, dim=1)
        )
        out = self.reduce_proxies(out, self.nb_proxy) if self.to_reduce else out
        out = self.sigma * out if self.sigma is not None else out

        return out  # [B, num_classes]
    
    def forward_all(self, input, task_id, inc=10, feature_dim=768):
        # input shape [batch, 768]
        # task class slice
        start_cls = task_id * inc  
        end_cls = start_cls + inc
        
        weight_task = self.weight[start_cls:end_cls, task_id*feature_dim:(task_id+1)*feature_dim]

        # cosine similarity
        out = F.linear(
            F.normalize(input, p=2, dim=1), 
            F.normalize(weight_task, p=2, dim=1)
        )
        out = self.reduce_proxies(out, self.nb_proxy) if self.to_reduce else out
        out = self.sigma * out if self.sigma is not None else out

        return out  # [B, num_classes]
    
    def forward_diagonal(self, input, task_id, inc=10, feature_dim=768):
        # input shape [batch, 768*num_task]
        for i in range(task_id + 1):
            start_cls = i * inc
            end_cls = start_cls + inc

            input_s = input[:, i * feature_dim : (i+1) * feature_dim]
            weight_s = self.weight[start_cls : end_cls, i * feature_dim : (i+1) * feature_dim]

            out = F.linear(
                F.normalize(input_s, p=2, dim=1), 
                F.normalize(weight_s, p=2, dim=1)
            )

            if i == 0:
                out_all = out
            else: 
                out_all = torch.cat((out_all, out), dim=1)
        
        out = self.reduce_proxies(out, self.nb_proxy) if self.to_reduce else out
        out = self.sigma * out if self.sigma is not None else out

        return out