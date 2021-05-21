#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from crypten.cuda import CUDALongTensor


def generate_random_ring_element(size, ring_size=(2 ** 64), device='cpu', **kwargs):
    """Helper function to generate a random number from a signed ring"""
    # TODO (brianknott): Check whether this RNG contains the full range we want.
    # device = "cpu" if "device" not in kwargs else kwargs["device"]
    gen = kwargs["generator"] if "generator" in kwargs else None
    rand_element = torch.empty(size=size, dtype=torch.long, device=device).random_(-(ring_size // 2), to=(ring_size - 1) // 2, generator=gen)
    
    if rand_element.is_cuda:
        return CUDALongTensor(rand_element)
    return rand_element


def generate_kbit_random_tensor(size, bitlength=None, device='cpu', **kwargs):
    """Helper function to generate a random k-bit number"""
    if bitlength is None:
        bitlength = torch.iinfo(torch.long).bits
    if bitlength == 64:
        return generate_random_ring_element(size, device=device, **kwargs)
    rand_tensor = torch.randint(0, 2 ** bitlength, size, dtype=torch.long, device=device, **kwargs)
    if rand_tensor.is_cuda:
        return CUDALongTensor(rand_tensor)
    return rand_tensor
