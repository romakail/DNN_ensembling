import os
import torch
from copy import deepcopy

class TwoModelsMSE ():
    def __init__(self, orig_model, weight_decay):
        self.orig_model = deepcopy(orig_model)
        for p in self.orig_model.parameters():
            p = p.detach()
        self.wd = weight_decay
        print (self.wd)
        
    def reg (self, model):
        l2 = 0.0
        x_sum = 0.
        y_sum = 0.
        for (x, y) in zip(self.orig_model.parameters(), model.parameters()):
            x_sum += x.sum().item()
            y_sum += y.sum().item()
#             print(torch.sqrt(torch.sum((x - y) ** 2)))
#             l2 += torch.sqrt(torch.sum((x - y) ** 2))
            l2 += torch.sum((x - y) ** 2)
#         print ('X : ', x_sum, 'Y : ', y_sum)
# #         print ('regularized :, - self.wd * l2)
        return - self.wd * l2