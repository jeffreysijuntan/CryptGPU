#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import functools

import numpy as np
import torch
import torch.nn.functional as F
import crypten.communicator as comm
from crypten.cuda import CUDALongTensor


class ConfigBase(abc.ABC):
    def __init__(self, config, *args):
        self.config = config
        assert len(args) % 2 == 0, "Uneven number of configuration params."
        self.params = args[::2]
        self.values = args[1::2]

    def __enter__(self):
        self.old_values = []
        for p, v in zip(self.params, self.values):
            self.old_values.append(getattr(self.config, p))
            setattr(self.config, p, v)

    def __exit__(self, exc_type, exc_value, tb):
        for p, v in zip(self.params, self.old_values):
            setattr(self.config, p, v)
        return exc_type is None


def count_wraps(share_list):
    """Computes the number of overflows or underflows in a set of shares

    We compute this by counting the number of overflows and underflows as we
    traverse the list of shares.
    """
    result = torch.zeros_like(share_list[0], dtype=torch.long)
    prev = share_list[0]
    for cur in share_list[1:]:
        next = cur + prev
        result -= ((prev < 0) & (cur < 0) & (next > 0)).long()  # underflow
        result += ((prev > 0) & (cur > 0) & (next < 0)).long()  # overflow
        prev = next
    return result


def pool_reshape(input, kernel_size, padding=None, stride=None, pad_value=0):
    """Rearrange a 4-d tensor so that each kernel is represented by each row"""
    # Setup kernel / stride values
    k = kernel_size
    if isinstance(k, int):
        k = (k, k)

    s = stride
    if s is None:
        s = k
    elif isinstance(s, int):
        s = (s, s)

    # Assert input parameters are correct type / size
    assert isinstance(k, tuple), "kernel_size must be an int or tuple"
    assert isinstance(s, tuple), "stride must be and int, a tuple, or None"
    assert len(k) == 2, "kernel_size must be an int or tuple pair"
    assert len(s) == 2, "stride must be an int or tuple pair"
    assert isinstance(pad_value, int), "pad_value must be an integer"
    assert input.dim() == 4, "pool input must be a 4-d tensor"

    # Apply padding if necessary
    if padding is not None:
        padding = (padding, padding) if isinstance(padding, int) else padding
        assert len(padding) == 2, "Padding must be an integer or a pair"
        padding = (padding[0], padding[0], padding[1], padding[1])
        input = torch.nn.functional.pad(input, padding, value=pad_value)

    # Compute output size based on parameters
    n = input.size(0)
    c = input.size(1)
    h = (input.size(2) - k[0]) // s[0] + 1
    w = (input.size(3) - k[1]) // s[1] + 1
    out_size = (n, c, h, w)

    # Reshape input to arrange kernels to be represented by rows
    kernel_indices = torch.tensor(range(k[1]), device=input.device)
    kernel_indices = torch.cat(
        [kernel_indices + i * input.size(3) for i in range(k[0])]
    )
    kernel_indices = torch.stack([kernel_indices + i * s[0] for i in range(w)])

    offset = input.size(3)
    kernel_indices = torch.cat([kernel_indices + i * s[1] * offset for i in range(h)])

    offset *= input.size(2)
    kernel_indices = torch.stack(
        [kernel_indices + i * offset for i in range(input.size(1))]
    )

    offset *= input.size(1)
    kernel_indices = torch.stack(
        [kernel_indices + i * offset for i in range(input.size(0))]
    )

    input = input.take(kernel_indices)

    return input, out_size


@functools.lru_cache(maxsize=10)
def chebyshev_series(func, width, terms):
    r"""Computes Chebyshev coefficients

    For n = terms, the ith Chebyshev series coefficient is

    .. math::
        c_i = 2/n \sum_{k=1}^n \cos(j(2k-1)\pi / 4n) f(w\cos((2k-1)\pi / 4n))

    Args:
        func (function): function to be approximated
        width (int): approximation will support inputs in range [-width, width]
        terms (int): number of Chebyshev terms used in approximation

    Returns:
        Chebyshev coefficients with shape equal to num of terms.
    """
    n_range = torch.arange(start=0, end=terms).float()
    x = width * torch.cos((n_range + 0.5) * np.pi / terms)
    y = func(x)
    cos_term = torch.cos(torch.ger(n_range, n_range + 0.5) * np.pi / terms)
    coeffs = (2 / terms) * torch.sum(y * cos_term, axis=1)
    return coeffs


# FIXME: pytorch currently does not register `torch.cat` and
# `torch.stack` in __torch_function__. We therefore can not call
# torch.stack/torch.cat with CUDALongTensor as parameters. This is
# a temporary solution before pytorch fix their issue.
# See https://github.com/pytorch/pytorch/issues/34294 for details
def torch_cat(tensors, dim=0, out=None):
    is_cuda = any(t.is_cuda for t in tensors)
    if is_cuda:
        return CUDALongTensor.cat(tensors, dim=dim, out=out)
    return torch.cat(tensors, dim=dim, out=out)


def torch_stack(tensors, dim=0, out=None):
    is_cuda = any(t.is_cuda for t in tensors)
    if is_cuda:
        return CUDALongTensor.stack(tensors, dim=dim, out=out)
    return torch.stack(tensors, dim=dim, out=out)


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
  # First figure out what the size of the output should be
  N, C, H, W = x_shape
#   assert (H + 2 * padding - field_height) % stride == 0
#   assert (W + 2 * padding - field_width) % stride == 0
  out_height = (H + 2 * padding - field_height) // stride + 1
  out_width = (W + 2 * padding - field_width) // stride + 1

  i0 = np.repeat(np.arange(field_height), field_width)
  i0 = np.tile(i0, C)
  i1 = stride * np.repeat(np.arange(out_height), out_width)
  j0 = np.tile(np.arange(field_width), field_height * C)
  j1 = stride * np.tile(np.arange(out_width), out_height)
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)

  k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

  return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
  """ An implementation of im2col based on some fancy indexing """
  # Zero-pad the input
  padding = padding[0] if isinstance(padding, tuple) else padding
  stride = stride[0] if isinstance(stride, tuple) else stride

  p = padding
  x_padded = F.pad(x.share, (p, p, p, p), mode='constant')

  k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                               stride)

  cols = x_padded[:, k, i, j]
  C = x.shape[1]
  cols = cols.permute(1, 2, 0).reshape(field_height * field_width * C, -1)

  from crypten.mpc import MPCTensor
  return MPCTensor.from_shares(cols, src=comm.get().get_rank())


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
  """ An implementation of col2im based on fancy indexing and np.add.at """
  padding = padding[0] if isinstance(padding, tuple) else padding
  stride = stride[0] if isinstance(stride, tuple) else stride

  cols = cols.share

  N, C, H, W = x_shape
  H_padded, W_padded = H + 2 * padding, W + 2 * padding

  from crypten.cuda import CUDALongTensor
  x_padded = torch.zeros((N, C, H_padded, W_padded), dtype=cols.dtype, device=cols.device)
  if x_padded.device.type == 'cuda':
      x_padded = CUDALongTensor(x_padded)

  k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                               stride)
  cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
  cols_reshaped = cols_reshaped.permute(2, 0, 1)

  k, i, j = torch.tensor([k], device=cols.device), torch.tensor([i], device=cols.device), torch.tensor([j], device=cols.device)
  l = torch.arange(N, device=cols.device).unsqueeze_(1).repeat(1,k.shape[0]).unsqueeze_(2)  
  x_padded.index_put_((l, k, i, j), cols_reshaped, accumulate=True)

  from crypten.mpc import MPCTensor
  x_padded = MPCTensor.from_shares(x_padded, src=comm.get().get_rank())

  if padding == 0:
    return x_padded
  return x_padded[:, :, padding:-padding, padding:-padding]