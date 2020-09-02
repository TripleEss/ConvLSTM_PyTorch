import argparse

import numpy as np
import torch
from PIL import Image

from moving_mnist_dataset import MovingMnistDataset
from seq2seq import ConvLSTMEncoderPredictor


def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', '-m', type=str, default=None)
    parser.add_argument('--id', '-i', type=int, default=0)
    args = parser.parse_args()

    test = MovingMnistDataset(phase_train=False)

    model = ConvLSTMEncoderPredictor(image_size=(64, 64))

    if args.model_path is not None:
        print("loading model from " + args.model_path)
        model.load_state_dict(torch.load(args.model_path))

    data, target = test[args.id]

    data = np.expand_dims(data, 0)
    target = np.expand_dims(target, 0)

    data = torch.from_numpy(data.astype(np.float32)).clone()
    res = model(data).to('cpu').detach().numpy().copy()


if __name__ == '__main__':
    inference()
