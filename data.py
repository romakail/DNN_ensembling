import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

class Transforms:

    class CIFAR10:

        class VGG:

            train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        class ResNet:

            train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

    CIFAR100 = CIFAR10

class GradBoostDataset (Dataset):
    def __init__(self, inputs_tensor, targets_tensor, logits_tensor, transform=None):
        self.inputs  = inputs_tensor
        self.logits  = logits_tensor
        self.targets = targets_tensor
        self.transform = transform
        assert len(inputs_tensor) == len(logits_tensor)
        assert len(logits_tensor) == len(targets_tensor)
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
#         print ("Inputs :", type(self.inputs[idx]))
#         print ("Target :", type(self.targets[idx]))
#         print ("Logits :", type(self.logits[idx]))
        
        img, target, label = self.inputs[idx], self.targets[idx], self.logits[idx]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, target, label
    
    def update_logits(self, logits_generator=None):
        if logits_generator is not None:
            self.logits += logits_generator(self.inputs)
        print ('Shape :', self.logits.shape, 'Logits_mean :', self.logits.mean().item())
        print ('Max :'  , self.logits.max().item(), 'Min :' , self.logits.min().item())

def loaders(dataset, path, batch_size, num_workers, transform_name, use_test=False,
            shuffle_train=True, weights_generator=None, logits_generator=None):
#     print ("Weights_generator", weights_generator, "logits_generator", logits_generator)
    ds = getattr(torchvision.datasets, dataset)
    path = os.path.join(path, dataset.lower())
    transform = getattr(getattr(Transforms, dataset), transform_name)
    train_set = ds(path, train=True, download=True, transform=transform.train)

    n_classes = max(train_set.targets) + 1
    if   weights_generator is not None and logits_generator is     None:
        weights = weights_generator (train_set)
        train_set.targets = [{'label': label, 'weight': weight.item()} for label, weight in zip(train_set.targets, weights)]
        
    elif weights_generator is     None and logits_generator is not None:
        logits = logits_generator(train_set.data).cpu().detach()
        print ('Initial logits :')
        print ('Shape :', logits.shape, 'Logits_mean :', logits.mean().item())
        print ('Max :'  , logits.max().item(), 'Min :' , logits.min().item())
        train_set = GradBoostDataset(
            train_set.data, 
            train_set.targets,
            logits,
            transform=transform.train)
        
    if use_test:
        print('You are going to run models on the test set. Are you sure?')
        test_set = ds(path, train=False, download=True, transform=transform.test)
    else:
        print("Using train (45000) + validation (5000)")
#         print (vars(train_set).keys())
#         print (train_set.data.shape)
#         print (len(train_set.targets))

        train_set.data = train_set.data[:-5000]
        train_set.targets = train_set.targets[:-5000]

        test_set = ds(path, train=True, download=True, transform=transform.test)
        test_set.train = False
        test_set.data = test_set.data[-5000:]
        test_set.targets = test_set.targets[-5000:]

    return {
               'train': torch.utils.data.DataLoader(
                   train_set,
                   batch_size=batch_size,
                   shuffle=shuffle_train,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'test': torch.utils.data.DataLoader(
                   test_set,
                   batch_size=batch_size,
                   shuffle=False,
                   num_workers=num_workers,
                   pin_memory=True
               ),
           }, n_classes
