#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Ngoc Nghia - Nghiann3
Created Date: Fri March 12 13:00:00 VNT 2021
Project : AkaOCR core
_____________________________________________________________________________

This file contains EfficientNet backbone
_____________________________________________________________________________
"""
import torch
from torch import nn
from torch.nn import functional as F
from models.modules.utils import (
    round_filters,
    round_repeats,
    drop_connect,
    Conv2dStaticSamePadding,
    get_model_params,
    efficientnet_params,
    Swish,
    MemoryEfficientSwish,
)

class EfficientNet(nn.Module):
    """
    EfficientNet model
    Args:
        compound_coef: to calculate width, depth, res, dropout of model
    """
    def __init__(self, compound_coef):
        super(EfficientNet, self).__init__()

        # blocks_args (list): A list of BlockArgs to construct blocks     
        # global_params (namedtuple): A set of GlobalParams shared between blocks
        self.blocks_args, self.global_params = get_model_params(compound_coef)   
        assert isinstance(self.blocks_args, list), 'blocks_args should be a list'
        assert len(self.blocks_args) > 0, 'block args must be greater than 0'
        # Get static or dynamic convolution depending on image size
        Conv2d = Conv2dStaticSamePadding

        # Batch norm parameters
        bn_mom = 1 - self.global_params.batch_norm_momentum
        bn_eps = self.global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self.global_params)  # number of output channels
        self.conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self.blocks = nn.ModuleList([])
        for block_args in self.blocks_args:
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self.global_params),
                output_filters=round_filters(block_args.output_filters, self.global_params),
                num_repeat=round_repeats(block_args.num_repeat, self.global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self.blocks.append(MBConvBlock(block_args, self.global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self.blocks.append(MBConvBlock(block_args, self.global_params))
        self.swish = MemoryEfficientSwish()

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn0(x)
        x = self.swish(x)
        feature_maps = []

        last_x = None
        for idx, block in enumerate(self.blocks):
            drop_connect_rate = self.global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if block.depthwise_conv.stride == [2, 2]:
                feature_maps.append(last_x)
            elif idx == len(self.blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x
        del x
        return feature_maps

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self.block_args = block_args
        self.bn_mom = 1 - global_params.batch_norm_momentum
        self.bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self.block_args.se_ratio is not None) and (0 < self.block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = Conv2dStaticSamePadding

        # Expansion phase
        inp = self.block_args.input_filters  # number of input channels
        oup = self.block_args.input_filters * self.block_args.expand_ratio  # number of output channels
        if self.block_args.expand_ratio != 1:
            self.expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self.bn0 = nn.BatchNorm2d(num_features=oup, momentum=self.bn_mom, eps=self.bn_eps)

        # Depthwise convolution phase
        k = self.block_args.kernel_size
        s = self.block_args.stride
        self.depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=oup, momentum=self.bn_mom, eps=self.bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self.block_args.input_filters * self.block_args.se_ratio))
            self.se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self.se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self.block_args.output_filters
        self.project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self.bn_mom, eps=self.bn_eps)
        self.swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self.block_args.expand_ratio != 1:
            x = self.expand_conv(inputs)
            x = self.bn0(x)
            x = self.swish(x)

        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self.se_reduce(x_squeezed)
            x_squeezed = self.swish(x_squeezed)
            x_squeezed = self.se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        x = self.project_conv(x)
        x = self.bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self.block_args.input_filters, self.block_args.output_filters
        if self.id_skip and self.block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self.swish = MemoryEfficientSwish() if memory_efficient else Swish()