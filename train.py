import argparse
import datetime
import os
import queue
import sys
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


def eval_net(net, criterion, optimizer,
             valid_loader, device):
    net.eval()
    running_loss = 0

    for _, (x, y) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        x = x.to(device)
        y = y.to(device).view(-1)

        with torch.no_grad():
            y_pred = net(x).view(-1)

        running_loss += criterion(y_pred, y).item()

    valid_loss = running_loss / len(valid_loader)
    return valid_loss


def log(log_dir, now, epoch, train_loss, valid_loss, elapsed_time):
    # Make dir
    os.makedirs(log_dir, exist_ok=True)

    if os.path.exists(os.path.join(log_dir, "log.csv")):
        # 過去のログ読み込み
        log_df = pd.read_csv(os.path.join(log_dir, "log.csv"))
    else:
        # 新規ログファイル作成
        log_df = pd.DataFrame(
            [], columns=["datetime", "epoch", "train_loss", "valid_loss", "elapsed_time"])

    tmp_log = pd.DataFrame([[now, epoch, train_loss, valid_loss, elapsed_time]],
                           columns=["datetime", "epoch", "train_loss", "valid_loss", "elapsed_time"])
    log_df = pd.concat([log_df, tmp_log])
    log_df.to_csv(os.path.join(log_dir, "log.csv"), index=False)
    print("Save log : " + os.path.join(log_dir, "log.csv"))


class ModelSaver:
    def __init__(self, net, optimizer, scheduler=None, save_dir=None, n_saved=1):
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.n_saved = n_saved

        self.reset()

    def reset(self):
        # queue to remember metric
        self._q = queue.Queue()
        for _ in range(self.n_saved):
            self._q.put(None)

        self._prev_metric = sys.float_info.max

    def save(self, epoch, now, metric):
        if self.save_dir is None:
            return

        # Make dir
        os.makedirs(self.save_dir, exist_ok=True)

        if metric <= self._prev_metric:
            # Save weight
            weight_file_name = os.path.join(
                self.save_dir, f"weight_epoch_{epoch}_{now}.pth")
            torch.save(self.net.state_dict(), weight_file_name)
            print("Save model:", weight_file_name)

            # Lifecycle
            self._q.put(weight_file_name)
            delete_file_name = self._q.get()
            if delete_file_name is not None:
                os.remove(delete_file_name)
            self._prev_metric = metric

        # Save optimizer
        optimizer_file_name = os.path.join(
            self.save_dir, "optimizer.pth")
        torch.save(self.optimizer.state_dict(), optimizer_file_name)
        print("Save optimizer :", optimizer_file_name)

        # Save scheduler
        if self.scheduler is not None:
            scheduler_file_name = os.path.join(
                self.save_dir, "scheduler.pth")
            torch.save(self.scheduler.state_dict(), scheduler_file_name)
            print("Save optimizer :", scheduler_file_name)


class EarlyStopper:
    def __init__(self, patience):
        self.patience = patience
        self._prev_metric = sys.float_info.max
        self._counter = 0

        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

    def judge(self, metric):
        if metric >= self._prev_metric:
            self._counter += 1
            if self._counter >= self.patience:
                print("EarlyStopping: Stop training")
                return True
        else:
            self._counter = 0
            self._prev_metric = metric
        return False


def run(exp_id, num_epochs, net,
        criterion, optimizer, scheduler,
        train_loader, valid_loader, device,
        log_dir, save_dir, n_saved, es_patience):

    # check parameters
    assert exp_id is not None
    assert net is not None
    assert criterion is not None
    assert optimizer is not None
    assert train_loader is not None
    assert valid_loader is not None
    assert device is not None

    # logging loss
    history = {'train_loss': [], 'valid_loss': []}

    saver = ModelSaver(net, optimizer, scheduler,
                       os.path.join(save_dir, exp_id), n_saved)

    es = EarlyStopper(es_patience)

    for epoch in range(num_epochs):
        start = time.time()  # start a epoch ---

        train_loss = train(net, criterion, optimizer,
                           train_loader, device)
        history['train_loss'].append(train_loss)

        valid_loss = eval_net(net, criterion, optimizer,
                              valid_loader, device)
        history['valid_loss'].append(valid_loss)

        elapsed_time = time.time() - start  # end a epoch ---

        print(f"epoch:{epoch}",
              "--", "train loss:{:.5f}".format(train_loss),
              "--", "valid loss:{:.5f}".format(valid_loss),
              "--", "elapsed_time:{:.2f}".format(elapsed_time) + "[sec]", flush=True)

        now = datetime.datetime.today().strftime("%Y%m%d%H%M%S")

        log(log_dir, now, epoch, train_loss, valid_loss, elapsed_time)

        if save_dir is not None:
            saver.save(epoch, now, valid_loss)

        if es.judge(valid_loss):
            break


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
    parser.add_argument('--save-model-path', type=str, default='./checkpoints',
                        help='For Saving the current Model (default: ./checkpoints)')
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


def main(args=None):
    if args is None:
        args = argument_paser()

    # Set experiment id
    exp_id = str(uuid.uuid4())[:8] if args.exp_id is None else args.exp_id
    print(f'Experiment Id: {exp_id}', flush=True)

    # Fix seed
    torch.manual_seed(args.seed)

    # Configure gpu
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

    # Prepare model
    net = ConvLSTMEncoderPredictor(image_size=(64, 64)).to(device)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    criterion = nn.MSELoss()

    run(
        exp_id=exp_id,
        num_epochs=args.epochs,
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        log_dir=args.log_dir,
        save_dir=args.save_model_path,
        n_saved=args.n_saved,
        es_patience=args.es_patience
    )

    return exp_id, net


if __name__ == "__main__":
    main()

    # notebook
    # args = parser.parse_args(args=['--epochs', '3'])
