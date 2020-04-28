import os
import torch
import torchvision
import torchvision.transforms as transforms


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


def loaders(dataset, path, batch_size, num_workers, transform_name, use_test=False,
            shuffle_train=True, weights_generator=None):
    ds = getattr(torchvision.datasets, dataset)
    path = os.path.join(path, dataset.lower())
    transform = getattr(getattr(Transforms, dataset), transform_name)
    train_set = ds(path, train=True, download=True, transform=transform.train)

    n_classes = max(train_set.targets) + 1
    if weights_generator is not None:
        weights = weights_generator (train_set)
        train_set.targets = [{'label': label, 'weight': weight.item()} for label, weight in zip(train_set.targets, weights)]
    
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
