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



def linear_weigh_func(logits, labels, coef=1, normalize=False):
    assert logits.shape[0] == labels.shape[0]
    prediction = softmax(logits, dim=1)
    probas = prediction.take(
        (labels + torch.arange(0, prediction.numel(), prediction.shape[1])).cuda())
    w = coef * (1 - probas)
    if normalize:
        w /= torch.sum(w)
        w *= w.shape[0]
    return w

def exponential_weight_func(logits, labels, coef=1, normalize=False):
    prediction = softmax(logits, dim=1)
    probas = prediction.take(
        (labels + torch.arange(0, prediction.numel(), prediction.shape[1])).cuda())
    w = torch.exp(coef * (1 - probas))
    if normalize:
        w /= torch.sum(w)
        w *= w.shape[0]
    return w

def adalast_weight_func(logits, labels, coef=1, normalize=False):
    true_logits = logits.take(
        (labels + torch.arange(0, logits.numel(), logits.shape[1])).cuda())
    true_logits = true_logits * (logits.shape[1] - 1) / logits.shape[1] - logits.mean(dim=1)
#             print (labels.device)
#             print (logits.device)
    correct = 2 * torch.eq(labels.cuda(), logits.argmax(dim=1)) - 1
    w = torch.exp(- coef * true_logits * correct)
    if normalize:
        w /= torch.sum(w)
        w *= w.shape[0]
    return w

class AdaBoost ():
    def __init__(self):
        self.logits = None
    def weight_func(self, logits, labels, coef=1, normalize=False):
        if self.logits is None:
            self.logits = logits
        else:
            self.logits += logits
        true_logits = self.logits.take(
            (labels + torch.arange(0, self.logits.numel(), self.logits.shape[1])).cuda())
        true_logits = true_logits * (self.logits.shape[1] - 1) / self.logits.shape[1] - logits.mean(dim=1)
    #             print (label.device)
    #             print (self.logits.device)
        correct = 2 * torch.eq(labels.cuda(), self.logits.argmax(dim=1)) - 1
        w = torch.exp(- coef * true_logits * correct)
        if normalize:
            w /= torch.sum(w)
            w *= w.shape[0]
        return w



def dataset_weights_generator(model, coef, func_type=None, normalize=False, batch_size=64):
    print ('Func_type : ', '[', func_type, ']', sep='')
    if func_type == 'Lin':
        weight_func = linear_weigh_func
    elif func_type == 'Exp':
        weight_func = exponential_weight_func
    elif func_type == 'AdaLast':
        weight_func = adalast_weight_func
    elif func_type == 'AdaBoost':
        adaboost_container = AdaBoost()
        weight_func = adaboost_container.weight_func
    elif func_type == 'GradBoost':
        return None
    else:
        print ("I don't know this function, choose on of [Lin/Exp/AdaLast/AdaBoost]")
        return None

    def weights_generator(dataset):
        with torch.no_grad():
            input_batch = []
            # label_batch = []

            logits = []
            labels = []

            # true_labels = 0
            for idx, (input, label) in enumerate(dataset):
                input_batch.append(input.unsqueeze(0))
                # label_batch.append(label)
                labels.append(label)
                if (idx+1) % batch_size == 0:
                    input_batch = torch.cat(input_batch, dim=0)
                    logits_batch = model(input_batch.cuda())
                    # weights.append(weight_func(coef, logits, torch.tensor(label_batch)))
#                     true_labels += torch.eq(
#                         prediction.argmax(dim=1).cuda(),
#                         torch.tensor(label_batch).cuda()).sum().item()
                    logits.append(logits_batch)
                    input_batch = []

            input_batch = torch.cat(input_batch, dim=0)
            logits_batch = model(input_batch.cuda())
            logits.append(logits_batch)
            # weights.append(weight_func(coef, prediction, torch.tensor(label_batch)))
            # weights = torch.cat(weights, dim=0)

            logits = torch.cat(logits, dim=0)
            labels = torch.tensor(labels, dtype=torch.long)

            weights = weight_func(logits, labels, coef=coef, normalize=normalize)

            print ('Shape :', weights.shape, 'Weights_sum :', weights.sum().item())
            print ('Max :', weights.max().item(), 'Min :', weights.min().item())
        return weights

    return weights_generator
