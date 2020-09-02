"""
Pendding: pytorch-pfn-extras must more than 0.2.2
"""

import argparse
import datetime
import os
import time

import numpy as np
import pandas as pd
import pytorch_pfn_extras as ppe
import pytorch_pfn_extras.training.extensions as extensions
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Subset
from tqdm import tqdm as tqdm

from moving_mnist_dataset import MovingMnistDataset
from seq2seq import ConvLSTMEncoderPredictor


def train(net, criterion, optimizer, train_loader, device):
    net.train()
    running_loss = 0

    for _, (xx, yy) in tqdm(enumerate(train_loader), total=len(train_loader)):
        xx = xx.to(device)
        yy = yy.to(device).view(-1)

        y_pred = net(xx).view(-1)
        loss = criterion(y_pred, yy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    return train_loss


def train(manager, net, criterion, train_loader, device):
    while not manager.stop_trigger:
        net.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            with manager.run_iteration(step_optimizers=['main']):
                data, target = data.to(device), target.to(device)
                pred = net(data)
                loss = criterion(pred, target)
                ppe.reporting.report({'train/loss': loss.item()})
                loss.backward()


def eval_net(net, criterion, data, target, device):
    net.eval()
    running_loss = 0

    data = data.to(device)
    target = target.to(device).view(-1)

    with torch.no_grad():
        pred = net(data).view(-1)

    running_loss += criterion(pred, target).item()
    ppe.reporting.report({'val/loss': running_loss})


def argument_paser():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=16,
                        metavar='N',
                        help='input batch size for testing (default: 16)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--snapshot', type=str, default=None,
                        help='path to snapshot file')
    args = parser.parse_args()
    return args


def main():
    args = argument_paser()

    # Fix seed
    torch.manual_seed(77)

    # Config gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Prepare data
    dataset = MovingMnistDataset()
    train_index, valid_index = train_test_split(
        range(len(dataset)), test_size=0.3)
    train_loader = DataLoader(
        Subset(dataset, train_index), batch_size=args.batch_size, shuffle=True, **kwargs)
    valid_loader = DataLoader(
        Subset(dataset, valid_index), batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # Prepare model
    net = ConvLSTMEncoderPredictor(image_size=(64, 64)).to(device)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    criterion = nn.MSELoss()

    # manager.extend(...) also works
    my_extensions = [
        extensions.LogReport(),
        extensions.ProgressBar(),
        extensions.observe_lr(optimizer=optimizer),
        extensions.ParameterStatistics(net, prefix='model'),
        extensions.VariableStatisticsPlot(net),
        extensions.Evaluator(
            valid_loader, net,
            eval_func=lambda data, target:
                eval_net(net, criterion, data, target, device),
            progress_bar=True),
        extensions.PlotReport(
            ['train/loss', 'val/loss'], 'epoch', filename='loss.png'),
        extensions.PrintReport(['epoch', 'iteration',
                                'train/loss', 'val/loss', 'lr']),
        extensions.snapshot(),
    ]

    # Custom stop triggers can be added to the manager and
    # their status accessed through `manager.stop_trigger`
    trigger = None
    # trigger = ppe.training.triggers.EarlyStoppingTrigger(
    #     check_trigger=(1, 'epoch'), monitor='val/loss')

    # Define manager
    manager = ppe.training.ExtensionsManager(
        net, optimizer, args.epochs,
        extensions=my_extensions,
        iters_per_epoch=len(train_loader),
        stop_trigger=trigger)

    # Lets load the snapshot
    if args.snapshot is not None:
        state = torch.load(args.snapshot)
        manager.load_state_dict(state)

    # Execute train
    train(manager, net, criterion, train_loader, device)
    # Test function is called from the evaluator extension
    # to get access to the reporter and other facilities
    # test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(net.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
