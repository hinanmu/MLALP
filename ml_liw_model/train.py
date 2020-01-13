#@Time      :2019/12/22 0:30
#@Author    :zhounan
#@FileName  :train.py
import argparse
import torchvision.models as models
from ml_gcn_model.util import *
import torch
import torch.optim as optim
from ml_liw_model.models import Inceptionv3Rank
from ml_liw_model.voc import Voc2007Classification, Voc2012Classification
import torchvision.transforms as transforms
import os
import shutil

parser = argparse.ArgumentParser(description='VOC')
parser.add_argument('--data', default='../data/voc2007', type=str,
                    help='path to dataset (e.g. data/')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--image_size', default=299, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def instance_wise_loss(output, y):
    y_i = torch.eq(y, torch.ones_like(y))
    y_not_i = torch.eq(y, -torch.ones_like(y))

    column = torch.unsqueeze(y_i, 2)
    row = torch.unsqueeze(y_not_i, 1)
    truth_matrix = column * row
    column = torch.unsqueeze(output, 2)
    row = torch.unsqueeze(output, 1)
    sub_matrix = column - row
    exp_matrix = torch.exp(-sub_matrix)
    sparse_matrix = exp_matrix * truth_matrix
    sums = torch.sum(sparse_matrix, (1, 2))
    y_i_sizes = torch.sum(y_i, 1)
    y_i_bar_sizes = torch.sum(y_not_i, 1)
    normalizers = y_i_sizes * y_i_bar_sizes
    normalizers_zero = torch.logical_not(torch.eq(normalizers, torch.zeros_like(normalizers)))
    normalizers = normalizers[normalizers_zero]
    sums = sums[normalizers_zero]
    loss = sums / normalizers
    loss = torch.sum(loss)
    return loss

def label_wise_loss(output, y):
    output = torch.transpose(output, 0, 1)
    y = torch.transpose(y,0, 1)
    return instance_wise_loss(output, y)

def criterion(output, y):
    loss = 0.5 * instance_wise_loss(output, y) + label_wise_loss(output, y)
    return loss

def save_checkpoint(model, is_best, best_score, save_model_path, filename='checkpoint.pth.tar'):
    filename_ = filename
    filename = os.path.join(save_model_path, filename_)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    print('save model {filename}'.format(filename=filename))
    torch.save(model.state_dict(), filename)
    if is_best:
        filename_best = 'model_best.pth.tar'
        filename_best = os.path.join(save_model_path, filename_best)
        shutil.copyfile(filename, filename_best)

        filename_best = os.path.join(save_model_path, 'model_best_{score:.4f}.pth.tar'.format(score=best_score))
        shutil.copyfile(filename, filename_best)

def train(model, epoch, optimizer, train_loader):
    total_loss = 0
    total_size = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data[0].cuda(), target.cuda()
        else:
            data = data[0].cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()
        total_size += data.size(0)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss / total_size))

def test(model, test_loader):
    from utils import evaluate_metrics
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for data, target in test_loader:
            if use_cuda:
                data, target = data[0].cuda(), target.cuda()
            else:
                data = data[0].cuda()
            output = model(data)
            test_loss += criterion(output, target).item()
            outputs.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())

    outputs = np.asarray(outputs)
    targets = np.asarray(targets)
    targets[targets==-1] = 0
    pred = outputs.copy()
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    metrics = evaluate_metrics.evaluate(targets, outputs, pred)
    print(metrics)
    return test_loss

def main_voc2007():
    torch.manual_seed(123)
    if use_cuda:
        torch.cuda.manual_seed_all(123)
    train_dataset = Voc2007Classification(args.data, 'train')
    val_dataset = Voc2007Classification(args.data, 'val')
    test_dataset = Voc2007Classification(args.data, 'test')
    data_transforms = transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor(),
    ])
    train_dataset.transform = data_transforms
    val_dataset.transform = data_transforms
    test_dataset.transform = data_transforms
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers)
    num_classes = 20
    model = models.inception_v3(pretrained=True)
    model.eval()
    model.aux_logit = False
    for param in model.parameters():
        param.requires_grad = False

    model = Inceptionv3Rank(model, num_classes)
    if use_cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.model.fc.parameters())
    # Use exponential decay for fine-tuning optimizer
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)

    best_loss = 1e5
    # Train
    for epoch in range(1, args.epochs + 1):
        train(model, epoch, optimizer, train_loader)
        val_loss = test(model, val_loader)
        scheduler.step(epoch)

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            test(model, test_loader)
        save_checkpoint(model, is_best, best_loss,
                        save_model_path='../checkpoint/mlliw/voc2007/',
                        filename='voc2007_checkpoint.pth.tar')

def main_voc2012():
    torch.manual_seed(123)
    if use_cuda:
        torch.cuda.manual_seed_all(123)

    train_dataset = Voc2012Classification('../data/voc2012', 'train')
    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])
    test_dataset = Voc2012Classification('../data/voc2012', 'val')
    data_transforms = transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor(),
    ])
    train_dataset.dataset.transform = data_transforms
    val_dataset.dataset.transform = data_transforms
    test_dataset.transform = data_transforms

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers)
    num_classes = 20
    model = models.inception_v3(pretrained=True)
    model.eval()
    model.aux_logit = False
    for param in model.parameters():
        param.requires_grad = False

    model = Inceptionv3Rank(model, num_classes)
    if use_cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.model.fc.parameters())
    # Use exponential decay for fine-tuning optimizer
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)

    best_loss = 1e5
    # Train
    for epoch in range(1, args.epochs + 1):
        train(model, epoch, optimizer, train_loader)
        val_loss = test(model, val_loader)
        scheduler.step(epoch)

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            test(model, test_loader)
        save_checkpoint(model, is_best, best_loss,
                        save_model_path='../checkpoint/mlliw/voc2012/',
                        filename='voc2012_checkpoint.pth.tar')
if __name__ == '__main__':
    main_voc2007()
    main_voc2012()