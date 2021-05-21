#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import crypten
import torch

from .module import Module


class _Loss(Module):
    """
    Base criterion class that mimics Pytorch's Loss.
    """

    def __init__(self, reduction="mean", skip_forward=False):
        super(_Loss, self).__init__()
        if reduction != "mean":
            raise NotImplementedError("reduction %s not supported")
        self.reduction = reduction
        self.skip_forward = skip_forward

    def forward(self, *args, **kwargs):
        raise NotImplementedError("forward not implemented")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __getattribute__(self, name):
        if name != "forward":
            return object.__getattribute__(self, name)

        def forward_function(*args, **kwargs):
            """Silently encrypt Torch tensors if needed."""
            if self.encrypted or any(
                isinstance(arg, crypten.CrypTensor) for arg in args
            ):
                args = list(args)
                for idx, arg in enumerate(args):
                    if torch.is_tensor(arg):
                        args[idx] = crypten.cryptensor(arg)
            return object.__getattribute__(self, name)(*tuple(args), **kwargs)

        return forward_function


class CrossEntropyLoss(_Loss):
    r"""
    Creates a criterion that measures cross-entropy loss between the
    prediction :math:`x` and the target :math:`y`. It is useful when
    training a classification problem with `C` classes.

    The prediction `x` is expected to contain raw, unnormalized scores for each class.

    The prediction `x` has to be a Tensor of size either :math:`(N, C)` or
    :math:`(N, C, d_1, d_2, ..., d_K)`, where :math:`N` is the size of the minibatch,
    and with :math:`K \geq 1` for the `K`-dimensional case (described later).

    This criterion expects a class index in the range :math:`[0, C-1]` as the
    target `y` for each value of a 1D tensor of size `N`.

    The loss can be described as:

    .. math::
        \text{loss}(x, class) = -\log \left(
        \frac{\exp(x[class])}{\sum_j \exp(x[j])} \right )
        = -x[class] + \log \left (\sum_j \exp(x[j]) \right)

    The losses are averaged across observations for each batch

    Can also be used for higher dimension inputs, such as 2D images, by providing
    an input of size :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`,
    where :math:`K` is the number of dimensions, and a target of appropriate shape.
    """  # noqa: W605

    def forward(self, x, y):
        assert x.size() == y.size(), "input and target must have the same size"
        return x.cross_entropy(y, skip_forward=self.skip_forward)