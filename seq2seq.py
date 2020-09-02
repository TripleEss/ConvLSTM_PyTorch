import torch
import torch.nn as nn

from conv_lstm import ConvLSTM


class ConvLSTMEncoderPredictor(nn.Module):
    def __init__(self, image_size):
        """ConvLSTM Encoder Predictor.

        Parameters
        ----------
        image_size: (int, int)
            Shape of image.
        """

        super().__init__()

        self.encoder_1 = ConvLSTM(
            in_channels=1, hidden_channels=32, kernel_size=3, stride=1, image_size=image_size)
        self.encoder_2 = ConvLSTM(
            in_channels=32, hidden_channels=32, kernel_size=3, stride=1, image_size=image_size)
        self.encoder_3 = ConvLSTM(
            in_channels=32, hidden_channels=32, kernel_size=3, stride=1, image_size=image_size)

        self.predictor_1 = ConvLSTM(
            in_channels=32, hidden_channels=32, kernel_size=3, stride=1, image_size=image_size)
        self.predictor_2 = ConvLSTM(
            in_channels=32, hidden_channels=32, kernel_size=3, stride=1, image_size=image_size)
        self.predictor_3 = ConvLSTM(
            in_channels=32, hidden_channels=32, kernel_size=3, stride=1, image_size=image_size)

        self.conv2d = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x, hidden_state_1 = self.encoder_1(x)
        x, hidden_state_2 = self.encoder_2(x)
        x, hidden_state_3 = self.encoder_3(x)

        x, _ = self.predictor_1(torch.zeros_like(x), hidden_state_1)
        x, _ = self.predictor_2(x, hidden_state_2)
        x, _ = self.predictor_3(x, hidden_state_3)

        seq_output = []
        for t in range(x.shape[1]):
            tmp = self.conv2d(x[:, t, :, :, :])
            seq_output.append(tmp)
        output = torch.stack(seq_output, 1)

        return output
