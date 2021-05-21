#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten.communicator as comm
import torch
from crypten.encoder import FixedPointEncoder

from ..ptype import ptype as Ptype
from . import beaver, resharing, circuit
from .arithmetic import ArithmeticSharedTensor
from .binary import BinarySharedTensor


def _A2B(arithmetic_tensor):
    assert comm.get().get_world_size() == 3
    rank = comm.get().get_rank()

    size = arithmetic_tensor.size()
    device = arithmetic_tensor.device

    z1, z2 = BinarySharedTensor.PRZS(size, device=device).share, BinarySharedTensor.PRZS(size, device=device).share

    x1, x2 = arithmetic_tensor.share, resharing.replicate_shares(arithmetic_tensor.share)

    if rank == 0:
        b1 = BinarySharedTensor.from_shares(z1 ^ (x1 + x2), src=rank)
        b2 = BinarySharedTensor.from_shares(z2, src=rank)
    elif rank == 1:
        b1 = BinarySharedTensor.from_shares(z1, src=rank)
        b2 = BinarySharedTensor.from_shares(z2 ^ x1, src=rank)
    else:
        b1 = BinarySharedTensor.from_shares(z1, src=rank)
        b2 = BinarySharedTensor.from_shares(z2, src=rank)

    binary_tensor = circuit.extract_msb(b1, b2)
    binary_tensor.encoder = arithmetic_tensor.encoder

    return binary_tensor
    

def convert(tensor, ptype, **kwargs):
    tensor_name = ptype.to_tensor()
    if isinstance(tensor, tensor_name):
        return tensor
    if isinstance(tensor, ArithmeticSharedTensor) and ptype == Ptype.binary:
        return _A2B(tensor)
    else:
        raise TypeError("Cannot convert %s to %s" % (type(tensor), ptype.__name__))


def get_msb(arithmetic_tensor):
    return _A2B(arithmetic_tensor)
