import argparse
import datetime
import os
import time
import uuid
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Subset
from tqdm import tqdm as tqdm

import catalyst
from catalyst.callbacks.checkpoint import CheckpointCallback
from catalyst.callbacks.misc import EarlyStoppingCallback
from catalyst.dl import SupervisedRunner
from moving_mnist_dataset import MovingMnistDataset
from seq2seq import ConvLSTMEncoderPredictor


def main(args=None):
    if args is None:
        args = argument_paser()

    # Set experiment id
    exp_id = str(uuid.uuid4())[:8] if args.exp_id is None else args.exp_id
    print(f'Experiment Id: {exp_id}', flush=True)

    # Fix seed
    torch.manual_seed(args.seed)

    # Config gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Prepare data
    dataset = MovingMnistDataset()
    train_index, valid_index = train_test_split(
        range(len(dataset)), test_size=0.3)
    train_loader = DataLoader(
        Subset(dataset, train_index), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(
        Subset(dataset, valid_index), batch_size=args.test_batch_size, shuffle=False)
    loaders = {"train": train_loader, "valid": valid_loader}

    model = ConvLSTMEncoderPredictor(image_size=(64, 64)).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    criterion = nn.MSELoss()

    runner = SupervisedRunner(device=catalyst.utils.get_device())
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        loaders=loaders,
        # model will be saved to {logdir}/checkpoints
        logdir=os.path.join(args.log_dir, exp_id),
        callbacks=[CheckpointCallback(save_n_best=args.n_saved),
                   EarlyStoppingCallback(patience=args.es_patience,
                                         metric="loss",
                                         minimize=True,)],
        num_epochs=args.epochs,
        main_metric="loss",
        minimize_metric=True,
        fp16=None,
        verbose=True
    )

    return exp_id, model


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
    # parser.add_argument('--save-model-path', type=str, default='./checkpoints',
    #                     help='For Saving the current Model (default: ./checkpoints)')
    parser.add_argument('--n-saved', type=int, default=1,
                        help='For Saving the current Model (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=0,
    #                     help='logging interval (default: 0)')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='path to snapshot file (default: ./logs)')
    parser.add_argument('--es-patience', type=int, default=10,
                        help='Early stop patience (default: 10)')
    parser.add_argument('--exp-id', type=str, default=None,
                        help='experiment id')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

    # notebook
    # args = parser.parse_args(args=['--epochs', '3'])
