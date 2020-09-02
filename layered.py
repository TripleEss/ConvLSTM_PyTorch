import torch
import torch.nn as nn

from conv_lstm import ConvLSTM


class ConvLSTMLayered(nn.Module):
    def __init__(self, image_size):
        """Multi layeres ConvLSTM.

        Parameters
        ----------
        image_size: (int, int)
            Shape of image.
        """

        super().__init__()
        self.conv_lstm_1 = ConvLSTM(
            in_channels=1, hidden_channels=32, kernel_size=3, stride=1, image_size=image_size)
        self.conv_lstm_2 = ConvLSTM(
            in_channels=32, hidden_channels=32, kernel_size=3, stride=1, image_size=image_size)
        self.conv_lstm_3 = ConvLSTM(
            in_channels=32, hidden_channels=32, kernel_size=3, stride=1, image_size=image_size)
        self.conv2d = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x, _ = self.conv_lstm_1(x)
        x, _ = self.conv_lstm_2(x)
        x, _ = self.conv_lstm_3(x)

        seq_output = []
        for t in range(x.shape[1]):
            tmp = self.conv2d(x[:, t, :, :, :])
            seq_output.append(tmp)
        output = torch.stack(seq_output, 1)

        return output
