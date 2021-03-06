import argparse
import numpy as np
import os
import random
import sys
import tabulate
import time
import torch
import torch.nn.functional as F

import data
import models
import utils
import regularization

parser = argparse.ArgumentParser(description='FGE training')

parser.add_argument('--dir', type=str, default='/tmp/fge/', metavar='DIR',
                    help='training directory (default: /tmp/fge)')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true',
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                    help='checkpoint to eval (default: None)')

parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--cycle', type=int, default=4, metavar='N',
                    help='number of epochs to train (default: 4)')
parser.add_argument('--lr_1', type=float, default=0.05, metavar='LR1',
                    help='initial learning rate (default: 0.05)')
parser.add_argument('--lr_2', type=float, default=0.0001, metavar='LR2',
                    help='initial learning rate (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--device', type=int, default=0, metavar='N',
                    help='number of device to train on (default: 0)')

parser.add_argument('--regularizer', type=str, default=None, metavar='REGULARIZER',
                    help='regularizer type (None/MSE2/MAE2) (default: None)')
parser.add_argument('--reg_wd', type=float, default=0, metavar='WD',
                    help='coefficient in regularizer between 2 networks (default: 0)')
parser.add_argument('--weighted_samples', type=str, default=None, metavar='WEIGHT',
                    help='method of weighting samples before crossentropy (Lin/Exp/AdaLast/AdaBoost) (default: None)')
parser.add_argument('--weight_coef', type=float, default=0, metavar='WD',
                    help='intensity of increasing of errors weights (default: 0)')
# parser.add_argument('--grad_boost', action='store_true',
#                     help='Enables gradient boosting algorithm over FGE (default=False)')
parser.add_argument('--version', type=str, default='classic', metavar='GB_VERSION',
                    help='Version of gradient boosting implementation (classic/simple).')
parser.add_argument('--boost_lr', type=str, default='auto', metavar='BOOST_LR',
                    help='boosting learning rate (auto/some float value)')
parser.add_argument('--scheduler', type=str, default='cyclic', metavar='SCHEDULER',
                    help='learning rate scheduler of every cycle (cyclic/linear/slide)')

parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: random)')

args = parser.parse_args()


assert args.cycle % 2 == 0, 'Cycle length should be even'

os.makedirs(args.dir, exist_ok=False)
with open(os.path.join(args.dir, 'fge.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

torch.backends.cudnn.benchmark = True
if args.seed == 0:
    args.seed = random.randint(0, 1000000)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = 'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(device)

architecture = getattr(models, args.model)

if (args.boost_lr == 'auto'):
    boost_lr = 1.0
else:
    boost_lr = float(args.boost_lr)

if   args.dataset == "CIFAR10":
    num_classes = 10
elif args.dataset == "CIFAR100":
    num_classes = 100
model = architecture.base(num_classes=num_classes, **architecture.kwargs)

if   args.version == 'classic':
    criterion = torch.nn.MSELoss(reduction='none')
elif args.version == 'simple':
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
else:
    raise AssertionError('I don`t know this implementation of gradient boosting')

if   args.scheduler == 'cyclic':
    scheduler = utils.cyclic_learning_rate
elif args.scheduler == 'linear':
    scheduler = utils.linear_learning_rate
elif args.scheduler == 'slide':
    scheduler = utils.slide_learning_rate
else:
    raise AssertionError('I don`t know such scheduler')

checkpoint = torch.load(args.ckpt)
# start_epoch = checkpoint['epoch'] + 1
start_epoch = checkpoint['epoch']
model.load_state_dict(checkpoint['model_state'])
model.cuda()

loaders, num_classes = data.loaders_gb(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    args.use_test,
    shuffle_train=True,
    logits_generator=regularization.dataset_logits_generator(
        model,
        transform=getattr(getattr(data.Transforms, args.dataset), args.transform).train,
        batch_size=args.batch_size),
)


optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr_1,
    momentum=args.momentum,
    weight_decay=args.wd
)
optimizer.load_state_dict(checkpoint['optimizer_state'])

# test_res = utils.test(loaders['test'], model, criterion)
# print ('Initial quality: ', test_res['accuracy'])

ensemble_size = 0
predictions_sum = np.zeros((len(loaders['test'].dataset), num_classes))

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_nll', 'te_loss', 'te_acc', 'ens_acc', 'time']

if args.regularizer is None:
    regularizer = None
elif args.regularizer == 'MSE2':
    regularizer = regularization.TwoModelsMSE(model, args.reg_wd).reg


utils.save_checkpoint(
    args.dir,
    start_epoch,
    name='fge',
    model_state=model.state_dict(),
    optimizer_state=optimizer.state_dict(),
    boost_weight=1.)

logits_sum = 0
for epoch in range(args.epochs):
    time_ep = time.time()
    lr_schedule = scheduler(epoch, args.cycle, args.lr_1, args.lr_2)
    
    train_res = utils.train_gb(
        loaders['train'],
        model,
        optimizer,
        criterion,
        lr_schedule=lr_schedule,
        regularizer=regularizer,
        gb_version=args.version,
        boost_lr=boost_lr)
    test_res = utils.test_gb(
        loaders['test'],
        model,
        criterion,
        boost_lr=boost_lr)
    time_ep = time.time() - time_ep

    ens_acc = None

    if (epoch + 1) % args.cycle == 0:
        if args.boost_lr == 'auto':
            boost_lr = regularization.adjust_boost_lr(loaders['train'], model)
        print ('Boost_lr : ', boost_lr)
        ensemble_size += 1
        logits, targets = utils.logits(loaders['test'], model)
        print ('Shapes :', logits.shape, targets.shape)
        logits_sum += boost_lr * logits
        regularization.logits_info(logits, logits_sum=logits_sum)
        ens_acc = 100.0 * np.mean(np.argmax(logits_sum, axis=1) == targets)

#     if (epoch + 1) % (args.cycle // 2) == 0:
        utils.save_checkpoint(
            args.dir,
            start_epoch + epoch,
            name='fge',
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            boost_weight=boost_lr
        )

#     if args.regularizer is not None and (epoch + 1) % (args.cycle) == 0:
#         regularizer = regularization.TwoModelsMSE(model, args.reg_wd).reg
#     if args.regularizer is not None and (epoch + 1) % (args.cycle // 2) == args.cycle // 2:
#         regularizer = None

    if args.weighted_samples is not None and (epoch + 1) % args.cycle == 0:
        loaders['train'].dataset.update_logits(
            boost_lr,
            logits_generator=regularization.dataset_logits_generator(
                model,
                transform=getattr(getattr(
                        data.Transforms,
                        args.datasest),
                    args.transform).train,
                batch_size = args.batch_size))
        loaders['test'].dataset.update_logits(
            boost_lr,
            logits_generator=regularization.dataset_logits_generator(
                model,
                transform=getattr(getattr(
                        data.Transforms,
                        args.datasest),
                    args.transform).test,
                batch_size = args.batch_size))
        
#         model = architecture.base(num_classes=num_classes, **architecture.kwargs)
#         model.cuda()
        
    values = [epoch, lr_schedule(1.0), train_res['loss'], train_res['accuracy'], test_res['nll'], test_res['loss'], test_res['accuracy'], ens_acc, time_ep]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)