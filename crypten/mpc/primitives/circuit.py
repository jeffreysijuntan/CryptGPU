#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

from . import resharing
import crypten.communicator as comm
from crypten.common.util import torch_stack
    

def __linear_circuit_block(x_block, y_block, encoder):
    from .binary import BinarySharedTensor
    from crypten.cuda import CUDALongTensor
    ci = torch_stack([torch.zeros_like(x_block), torch.ones_like(y_block)])

    for i in range(8):
        xi = (x_block >> i) & 1
        yi = (y_block >> i) & 1

        xi, yi = torch_stack([xi, xi]), torch_stack([yi,yi])
        si = xi ^ yi ^ ci
        ci = ci ^ resharing.AND_gate(xi ^ ci, yi ^ ci).share

    select_bits = torch.zeros_like(ci[0,0])
    for i in range(8):
        select_bits = resharing.AND_gate(select_bits ^ 1, ci[0,i]).share ^ resharing.AND_gate(select_bits, ci[1,i]).share
    sign_bits =  resharing.AND_gate(select_bits ^ 1, si[0,7]).share ^ resharing.AND_gate(select_bits, si[1,7]).share

    sign_bits = sign_bits.long()
    if sign_bits.is_cuda:
        sign_bits = CUDALongTensor(sign_bits)

    sign_bits = BinarySharedTensor.from_shares(sign_bits, src=comm.get().get_rank())
    sign_bits.encoder = encoder

    return sign_bits


def extract_msb(x, y):
    x_block = torch_stack(
        [((x.share >> (i*8)) & 255).byte() for i in range(8)]
    )
    y_block = torch_stack(
        [((y.share >> (i*8)) & 255).byte() for i in range(8)]
    )

    return __linear_circuit_block(x_block, y_block, x.encoder)

