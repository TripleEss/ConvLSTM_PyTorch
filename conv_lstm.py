"""
Copyright (c) 2020 Masafumi Abeta. All Rights Reserved.
Released under the MIT license
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, hidden_channels,
                 kernel_size, stride=1, image_size=None):
        """ConvLSTM cell.

        Parameters
        ----------
        in_channels: int
            Number of channels of input tensor.
        hidden_channels: int
            Number of channels of hidden state.
        kernel_size: int or (int, int)
            Size of the convolutional kernel.
        stride: int or (int, int)
            Stride of the convolution.
        image_size: (int, int)
            Shape of image.
        """

        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

        # No bias for hidden, since bias is included in observation convolution
        # Pad the hidden layer so that the input and output sizes are equal
        self.Wxi = Conv2dStaticSamePadding(
            self.in_channels, self.hidden_channels, self.kernel_size, self.stride, image_size=image_size)
        self.Whi = Conv2dStaticSamePadding(
            self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride, image_size=image_size, bias=False)
        self.Wxf = Conv2dStaticSamePadding(
            self.in_channels, self.hidden_channels, self.kernel_size, self.stride, image_size=image_size)
        self.Whf = Conv2dStaticSamePadding(
            self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride, image_size=image_size, bias=False)
        self.Wxg = Conv2dStaticSamePadding(
            self.in_channels, self.hidden_channels, self.kernel_size, self.stride, image_size=image_size)
        self.Whg = Conv2dStaticSamePadding(
            self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride, image_size=image_size, bias=False)
        self.Wxo = Conv2dStaticSamePadding(
            self.in_channels, self.hidden_channels, self.kernel_size, self.stride, image_size=image_size)
        self.Who = Conv2dStaticSamePadding(
            self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride, image_size=image_size, bias=False)

    def forward(self, x, hidden_state):
        """
        Parameters
        ----------
        x: torch.Tensor
            4-D Tensor of shape (b, c, h, w).
        hs: tuple
            Previous hidden state of shape (h_0, c_0).

        Returns
        -------
            h_next, c_next
        """

        h_prev, c_prev = hidden_state
        i = torch.sigmoid(self.Wxi(x) + self.Whi(h_prev))
        f = torch.sigmoid(self.Wxf(x) + self.Whf(h_prev))
        o = torch.sigmoid(self.Wxo(x) + self.Who(h_prev))
        g = torch.tanh(self.Wxg(x) + self.Whg(h_prev))

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels,
                 kernel_size, stride=1, image_size=None):
        """ConvLSTM.

        Parameters
        ----------
        in_channels: int
            Number of channels of input tensor.
        hidden_channels: int
            Number of channels of hidden state.
        kernel_size: int or (int, int)
            Size of the convolutional kernel.
        stride: int or (int, int)
            Stride of the convolution.
        image_size: (int, int)
            Shape of image.
        """

        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.image_size = image_size

        self.lstm_cell = ConvLSTMCell(
            self.in_channels, self.hidden_channels, self.kernel_size, self.stride, image_size=self.image_size)

    def forward(self, xs, hidden_state=None):
        """
        Parameters
        ----------
        xs: torch.Tensor
            5-D Tensor of shape (b, t, c, h, w).
        hs: list
            Previous hidden state of shape (h_0, c_0).

        Returns
        -------
            last_state_list, layer_output
        """

        batch_size, sequence_length, _, height, width = xs.size()

        if hidden_state is None:
            hidden_state = (torch.zeros(batch_size, self.hidden_channels, height, width, device=xs.device),
                            torch.zeros(batch_size, self.hidden_channels, height, width, device=xs.device))

        output_list = []
        for t in range(sequence_length):
            hidden_state = self.lstm_cell(xs[:, t, ...], hidden_state)
            h, _ = hidden_state
            output_list.append(h)

        output = torch.stack(output_list, dim=1)

        return output, hidden_state


class Conv2dStaticSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.

        # Copyright: lukemelas (github username)
        # Released under the MIT License <https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/LICENSE>
        # <https://github.com/lukemelas/EfficientNet-PyTorch/blob/4d63a1f77eb51a58d6807a384dda076808ec02c0/efficientnet_pytorch/utils.py>
    """

    # With the same calculation as Conv2dDynamicSamePadding
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [
            self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(
            image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] +
                    (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] +
                    (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d(
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride,
                     self.padding, self.dilation, self.groups)
        return x
