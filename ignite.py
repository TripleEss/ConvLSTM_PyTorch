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
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Subset
from tqdm import tqdm as tqdm

from ignite.contrib.handlers import ProgressBar
from ignite.engine import (Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.handlers import EarlyStopping, ModelCheckpoint, Timer
from ignite.metrics import Accuracy, Loss, RunningAverage
from moving_mnist_dataset import MovingMnistDataset
from seq2seq import ConvLSTMEncoderPredictor


def write_metrics(metrics, writer, timer, mode: str, epoch: int):
    """print metrics & write metrics to log"""
    avg_loss = metrics['mse']
    print(f"{mode} Results - Epoch: {epoch} -- Avg loss: {avg_loss:.5f} -- Elapsed time: {timer.value():.2f}")
    if writer is not None:
        writer.add_scalar(f"{mode}/avg_loss", avg_loss, epoch)


def score_function(engine):
    val_loss = engine.state.metrics['mse']
    return - val_loss


def _epoch(engine, event_name):
    return engine.state.epoch


def run(exp_id, epochs, model, criterion, optimizer, scheduler,
        train_loader, valid_loader, device, writer, log_interval,
        n_saved, save_dir, es_patience):

    # check parameters
    assert exp_id is not None
    assert model is not None
    assert criterion is not None
    assert optimizer is not None
    assert train_loader is not None
    assert valid_loader is not None
    assert device is not None
    assert save_dir is not None

    trainer = create_supervised_trainer(
        model, optimizer, criterion, device=device)
    evaluator = create_supervised_evaluator(
        model,
        metrics={'mse': Loss(criterion)},
        device=device
    )

    # # Timer
    timer = Timer(average=False)
    timer.attach(trainer,
                 start=Events.EPOCH_STARTED,
                 pause=Events.EPOCH_COMPLETED,
                 resume=Events.EPOCH_STARTED,
                 step=Events.EPOCH_COMPLETED)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names='all')

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        if log_interval > 0:
            i = (engine.state.iteration - 1) % len(train_loader) + 1
            if i % log_interval == 0:
                print(f"Epoch[{engine.state.epoch}] -- Iteration[{i}/{len(train_loader)}] -- "
                      f"Loss: {engine.state.output:.5f} -- Elapsed time: {timer.value():.2f}")
                if writer is not None:
                    writer.add_scalar("training/loss", engine.state.output,
                                      engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        write_metrics(metrics, writer, timer, 'Training', engine.state.epoch)

        if scheduler is not None:
            scheduler.step()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(valid_loader)
        metrics = evaluator.state.metrics
        write_metrics(metrics, writer, timer, 'Validation', engine.state.epoch)

        pbar.n = pbar.last_print_n = 0

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_optimizer(engine):
        # Save optimizer
        optimizer_file_name = os.path.join(
            save_dir, exp_id, 'optimizer.pth')
        torch.save(optimizer.state_dict(), optimizer_file_name)
        print("Save optimizer :", optimizer_file_name)

        # Save scheduler
        if scheduler is not None:
            scheduler_file_name = os.path.join(
                save_dir, exp_id, 'scheduler.pth')
            torch.save(scheduler.state_dict(), scheduler_file_name)
            print("Save optimizer :", scheduler_file_name)

    # # Checkpoint setting
    # {save_dir}/{exp_id}/best_mymodel_{engine.state.epoch}
    # n_saved 個までモデルを保持する
    handler = ModelCheckpoint(dirname=f'{save_dir}/{exp_id}', filename_prefix='best',
                              n_saved=n_saved, create_dir=True, global_step_transform=_epoch)
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              handler, {'mymodel': model})

    # # Early stopping
    handler = EarlyStopping(
        patience=es_patience, score_function=score_function, trainer=trainer)
    # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset)
    evaluator.add_event_handler(Events.COMPLETED, handler)

    trainer.run(train_loader, max_epochs=epochs)


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
    parser.add_argument('--log-interval', type=int, default=0,
                        help='logging interval (default: 0)')
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

    # Set logger
    log_writer = SummaryWriter(log_dir=os.path.join(
        args.log_dir, exp_id)) if args.log_dir is not None else None

    # Prepare data
    dataset = MovingMnistDataset()
    train_index, valid_index = train_test_split(
        range(len(dataset)), test_size=0.3)
    train_loader = DataLoader(
        Subset(dataset, train_index), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(
        Subset(dataset, valid_index), batch_size=args.test_batch_size, shuffle=False)

    # Prepare model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ConvLSTMEncoderPredictor(image_size=(64, 64)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    criterion = nn.MSELoss()

    run(
        exp_id=exp_id,
        epochs=args.epochs,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        writer=log_writer,
        log_interval=args.log_interval,
        n_saved=args.n_saved,
        save_dir=args.save_model_path,
        es_patience=args.es_patience
    )

    log_writer.close()

    return exp_id, model


if __name__ == "__main__":
    main()

    # notebook
    # args = parser.parse_args(args=['--epochs', '3'])
