import os
import torch
from copy import deepcopy
from tqdm import tqdm

from torch.nn.functional import softmax

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
            l2 += torch.sum((x - y) ** 2)
        return - self.wd * l2
    
def dataset_weights_generator(model, coef, func_type=None, batch_size=64):
    print ('Func_type : ', '[', func_type, ']', sep='')
    if func_type == 'Lin':
        def weight_func(coef, logits, label):
            prediction = softmax(logits, dim=1)
            probas = prediction.take(
                (label + torch.arange(0, prediction.numel(), prediction.shape[1])).cuda())
            return coef * (1 - probas)
    elif func_type == 'Exp':
        def weight_func(coef, logits, label):
            prediction = softmax(logits, dim=1)
            probas = prediction.take(
                (label + torch.arange(0, prediction.numel(), prediction.shape[1])).cuda())
            return torch.exp(coef * (1 - probas))
    elif func_type == 'AdaLast':
        def weight_func(coef, logits, label):
            true_logits = logits.take(
                (label + torch.arange(0, logits.numel(), logits.shape[1])).cuda())
            true_logits = true_logits * (logits.shape[1] - 1) / logits.shape[1] - logits.mean(dim=1)
#             print (label.device)
#             print (logits.device)
            correct = 2 * torch.eq(label.cuda(), logits.argmax(dim=1)) - 1
            return torch.exp(- coef * true_logits * correct)
    else:
        print ("I don't know this function, choose on of [Lin/Exp/AdaLast]")
        return None
    
    def weights_generator(dataset):
        with torch.no_grad():
            weights = []
            input_batch = []
            label_batch = []
            weights     = []

            true_labels = 0
            for idx, (input, label) in enumerate(dataset):
                input_batch.append(input.unsqueeze(0))
                label_batch.append(label)
                if (idx+1) % batch_size == 0:
                    input_batch = torch.cat(input_batch, dim=0)
                    logits = model(input_batch.cuda())
                    weights.append(weight_func(coef, logits, torch.tensor(label_batch)))
#                     true_labels += torch.eq(
#                         prediction.argmax(dim=1).cuda(),
#                         torch.tensor(label_batch).cuda()).sum().item()
    
                    input_batch = []
                    label_batch = []
            
            input_batch = torch.cat(input_batch, dim=0)
            prediction = softmax(model(input_batch.cuda()), dim=1)
            weights.append(weight_func(coef, prediction, torch.tensor(label_batch)))
            
            weights = torch.cat(weights, dim=0)
            print ('Shape :', weights.shape, 'Weights_sum :', weights.sum().item())
            print ('Max :', weights.max().item(), 'Min :', weights.min().item())
        return weights
    
    return weights_generator
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       