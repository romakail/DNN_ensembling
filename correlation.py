import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm_notebook as tqdm

def predictions_correlation(prediction1, prediction2):
    assert len(prediction1.shape) == 2
    assert prediction1.shape == prediction2.shape

    correlation = (
        ( (prediction1 - prediction1.mean(dim=0, keepdim=True))
        * (prediction2 - prediction2.mean(dim=0, keepdim=True))).mean(dim=0)
        / prediction1.var(dim=0).sqrt()
        / prediction2.var(dim=0).sqrt())
    return correlation.cpu().numpy()

def correlation_over_classes (model1, model2, dataloader, device=torch.device('cpu')):
    model1.eval().to(device)
    model2.eval().to(device)

    prediction1 = []
    prediction2 = []

    for input, _ in dataloader:
        prediction1.append(model1(input.detach().to(device)).detach().cpu())
        prediction2.append(model2(input.detach().to(device)).detach().cpu())

    prediction1 = torch.cat(prediction1, dim=0)
    prediction2 = torch.cat(prediction2, dim=0)

    return predictions_correlation(prediction1, prediction2)

def correlation_among_models (model_list, dataloader, device=torch.device('cpu'), one_hot=False):

    predictions = []
    for iter, model in tqdm(enumerate(model_list)):
        model.eval().to(device)
        predictions.append([])
        for input, _ in dataloader:
            predictions[iter].append(model(input.detach().to(device)).detach().cpu())
        predictions[iter] = torch.cat(predictions[iter], dim=0)

    if one_hot:
        for idx, prediction in enumerate(predictions):
            predictions[idx] = F.one_hot(prediction.argmax(dim=1)).float()

    n_model_list = len(model_list)
    cor_matrix = np.zeros([n_model_list, n_model_list], dtype=float)
    for i in range(n_model_list):
        for j in range(i, n_model_list):
            cor_matrix[i, j] = cor_matrix[j, i] = (
                predictions_correlation(predictions[i], predictions[j]).mean())
    return cor_matrix

