#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math

import crypten
import torch

def _compute_pairwise_comparisons_for_steps(input_tensor, dim):
    """
    Helper function that does pairwise comparisons by splitting input
    tensor for `steps` number of steps along dimension `dim`.
    """
    enc_tensor_reduced = input_tensor.clone()
    while enc_tensor_reduced.size(dim) > 1:
        m = enc_tensor_reduced.size(dim)
        x, y, remainder = enc_tensor_reduced.split([m // 2, m // 2, m % 2], dim=dim)
        #pairwise_max = crypten.where(x >= y, x, y)
        pairwise_max = x.pairwise_max(y) 

        ptx = x.get_plain_text()
        pm = pairwise_max.get_plain_text()

        enc_tensor_reduced = crypten.cat([pairwise_max, remainder], dim=dim)
    return enc_tensor_reduced


def _max_helper_log_reduction(enc_tensor, dim=None, keepdim=False):
    """Returns max along dim `dim` using the log_reduction algorithm"""
    if enc_tensor.dim() == 0:
        return enc_tensor
    input, dim_used = enc_tensor, dim
    if dim is None:
        dim_used = 0
        input = enc_tensor.flatten()
    n = input.size(dim_used)  # number of items in the dimension
    enc_max_vec = _compute_pairwise_comparisons_for_steps(input, dim_used)

    return enc_max_vec


def _max_helper_all_tree_reductions(enc_tensor, dim=None, method="log_reduction", keepdim=False):
    """
    Finds the max along `dim` using the specified reduction method. `method`
    can be one of [`log_reduction`, `double_log_reduction`, 'accelerated_cascade`]
    `log_reduction`: Uses O(n) comparisons and O(log n) rounds of communication
    `double_log_reduction`: Uses O(n loglog n) comparisons and O(loglog n) rounds
    of communication (Section 2.6.2 in https://folk.idi.ntnu.no/mlh/algkon/jaja.pdf)
    `accelerated_cascade`: Uses O(n) comparisons and O(loglog n) rounds of
    communication. (See Section 2.6.3 of https://folk.idi.ntnu.no/mlh/algkon/jaja.pdf)
    """
    assert method == "log_reduction"
    if method == "log_reduction":
        return _max_helper_log_reduction(enc_tensor, dim, keepdim)
