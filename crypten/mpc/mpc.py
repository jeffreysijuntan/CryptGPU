#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import wraps

import crypten
import torch
from crypten import communicator as comm
from crypten.common.tensor_types import is_tensor
from crypten.common.util import ConfigBase, pool_reshape, torch_cat, torch_stack
from crypten.cuda import CUDALongTensor

from ..cryptensor import CrypTensor
from ..encoder import FixedPointEncoder
from .max_helper import _max_helper_all_tree_reductions
from .primitives import resharing
from .primitives.binary import BinarySharedTensor
from .primitives.converters import convert, get_msb
from .ptype import ptype as Ptype


def mode(ptype, inplace=False):
    if inplace:

        def function_wrapper(func):
            # @wraps ensures docstrings are updated
            @wraps(func)
            def convert_wrapper(self, *args, **kwargs):
                self._tensor = convert(self._tensor, ptype)
                self.ptype = ptype
                self = func(self, *args, **kwargs)
                return self

            return convert_wrapper

    else:

        def function_wrapper(func):
            # @wraps ensures docstrings are updated
            @wraps(func)
            def convert_wrapper(self, *args, **kwargs):
                result = self.to(ptype)
                return func(result, *args, **kwargs)

            return convert_wrapper

    return function_wrapper


def _one_hot_to_index(tensor, dim, keepdim, device=None):
    """
    Converts a one-hot tensor output from an argmax / argmin function to a
    tensor containing indices from the input tensor from which the result of the
    argmax / argmin was obtained.
    """
    if tensor.ptype == Ptype.binary:
        tensor = tensor.to(Ptype.arithmetic, bits=1)
    #print(tensor.get_plain_text())

    if dim is None:
        result = tensor.flatten()
        result = result * torch.tensor(list(range(tensor.nelement())), device=device)
        return result.sum()
    else:
        size = [1] * tensor.dim()
        size[dim] = tensor.size(dim)
        result = tensor * torch.tensor(
            list(range(tensor.size(dim))), device=device
        ).view(size)
        return result.sum(dim, keepdim=keepdim)


@dataclass
class MPCConfig:
    """
    A configuration object for use by the MPCTensor.
    """

    # exponential function
    exp_iterations: int = 9

    # reciprocal configuration
    reciprocal_method: str = "NR"
    reciprocal_nr_iters: int = 13
    reciprocal_log_iters: int = 1
    reciprocal_all_pos: bool = False
    reciprocal_initial: any = 1/200

    # log configuration
    log_iterations: int = 2
    log_exp_iterations: int = 8
    log_order: int = 8

    # _eix configuration
    _eix_iterations: int = 10

    # Used by max / argmax / min / argmin
    max_method: str = "log_reduction"

    use_rss: bool = True


# Global config
config = MPCConfig()


class ConfigManager(ConfigBase):
    r"""
    Use this to temporarily change a value in the `mpc.config` object. The
    following sets `config.exp_iterations` to `10` for one function
    invocation and then sets it back to the previous value::

        with ConfigManager("exp_iterations", 10):
            tensor.exp()

    """

    def __init__(self, *args):
        super().__init__(config, *args)


class MPCTensor(CrypTensor):
    def __init__(self, tensor, ptype=Ptype.arithmetic, device=None, *args, **kwargs):
        # take required_grad from kwargs, input tensor, or set to False:
        default = tensor.requires_grad if torch.is_tensor(tensor) else False
        requires_grad = kwargs.pop("requires_grad", default)

        # call CrypTensor constructor:
        super().__init__(requires_grad=requires_grad)
        if tensor is None:
            return  # TODO: Can we remove this and do staticmethods differently?

        if device is None and hasattr(tensor, "device"):
            device = tensor.device

        # create the MPCTensor:
        tensor_type = ptype.to_tensor()
        self._tensor = tensor_type(tensor, device=device, *args, **kwargs)
        self.ptype = ptype

    @staticmethod
    def new(*args, **kwargs):
        """
        Creates a new MPCTensor, passing all args and kwargs into the constructor.
        """
        return MPCTensor(*args, **kwargs)

    @staticmethod
    def from_shares(share, precision=None, src=0, ptype=Ptype.arithmetic):
        result = MPCTensor(None)
        from_shares = ptype.to_tensor().from_shares
        result._tensor = from_shares(share, precision=precision, src=src)
        result.ptype = ptype
        return result

    def clone(self):
        """Create a deep copy of the input tensor."""
        # TODO: Rename this to __deepcopy__()?
        result = MPCTensor(None)
        result._tensor = self._tensor.clone()
        result.ptype = self.ptype
        return result

    def shallow_copy(self):
        """Create a shallow copy of the input tensor."""
        # TODO: Rename this to __copy__()?
        result = MPCTensor(None)
        result._tensor = self._tensor
        result.ptype = self.ptype
        return result

    def copy_(self, other):
        """Copies value of other MPCTensor into this MPCTensor."""
        assert isinstance(other, MPCTensor), "other must be MPCTensor"
        self._tensor.copy_(other._tensor)
        self.ptype = other.ptype

    def to(self, *args, **kwargs):
        r"""
        Depending on the input arguments,
        converts underlying share to the given ptype or
        performs `torch.to` on the underlying torch tensor

        To convert underlying share to the given ptype, call `to` as:
            to(ptype, **kwargs)

        It will call MPCTensor.to_ptype with the arguments provided above.

        Otherwise, `to` performs `torch.to` on the underlying
        torch tensor. See
        https://pytorch.org/docs/stable/tensors.html?highlight=#torch.Tensor.to
        for a reference of the parameters that can be passed in.

        Args:
            ptype: Ptype.arithmetic or Ptype.binary.
        """
        if "ptype" in kwargs:
            return self._to_ptype(**kwargs)
        elif args and isinstance(args[0], Ptype):
            ptype = args[0]
            return self._to_ptype(ptype, **kwargs)
        else:
            share = self.share.to(*args, **kwargs)
            if share.is_cuda:
                share = CUDALongTensor(share)
            self.share = share
            return self

    def _to_ptype(self, ptype, **kwargs):
        r"""
        Convert MPCTensor's underlying share to the corresponding ptype
        (ArithmeticSharedTensor, BinarySharedTensor)

        Args:
            ptype (Ptype.arithmetic or Ptype.binary): The ptype to convert
                the shares to.
            precision (int, optional): Precision of the fixed point encoder when
                converting a binary share to an arithmetic share. It will be ignored
                if the ptype doesn't match.
            bits (int, optional): If specified, will only preserve the bottom `bits` bits
                of a binary tensor when converting from a binary share to an arithmetic share.
                It will be ignored if the ptype doesn't match.
        """

        retval = self.clone()
        if retval.ptype == ptype:
            return retval
        retval._tensor = convert(self._tensor, ptype, **kwargs)
        retval.ptype = ptype
        return retval

    def arithmetic(self):
        """Converts self._tensor to arithmetic secret sharing"""
        return self.to(Ptype.arithmetic)

    def binary(self):
        """Converts self._tensor to binary secret sharing"""
        return self.to(Ptype.binary)

    @property
    def device(self):
        """Return the `torch.device` of the underlying share"""
        return self.share.device

    @property
    def is_cuda(self):
        """Return True if the underlying share is stored on GPU, False otherwise"""
        return self.share.is_cuda

    def cuda(self, *args, **kwargs):
        """Call `torch.Tensor.cuda` on the underlying share"""
        self.share = CUDALongTensor(self.share.cuda(*args, **kwargs))
        return self

    def cpu(self):
        """Call `torch.Tensor.cpu` on the underlying share"""
        self.share = self.share.cpu()
        return self

    def get_plain_text(self, dst=None):
        """Decrypts the tensor."""
        return self._tensor.get_plain_text(dst=dst)

    def reveal(self, dst=None):
        """Decrypts the tensor without any downscaling."""
        return self._tensor.reveal(dst=dst)

    def __bool__(self):
        """Override bool operator since encrypted tensors cannot evaluate"""
        raise RuntimeError("Cannot evaluate MPCTensors to boolean values")

    def __nonzero__(self):
        """__bool__ for backwards compatibility with Python 2"""
        raise RuntimeError("Cannot evaluate MPCTensors to boolean values")

    def __repr__(self):
        """Returns a representation of the tensor useful for debugging."""
        from crypten.debug import debug_mode

        if not hasattr(self, "_tensor"):
            return "MPCTensor(None)"
        share = self.share
        plain_text = self._tensor.get_plain_text() if debug_mode() else "HIDDEN"
        ptype = self.ptype
        return (
            f"MPCTensor(\n\t_tensor={share}\n"
            f"\tplain_text={plain_text}\n\tptype={ptype}\n)"
        )

    def __setitem__(self, index, value):
        """Set tensor values by index"""
        if not isinstance(value, MPCTensor):
            value = MPCTensor(value, ptype=self.ptype, device=self.device)
        self._tensor.__setitem__(index, value._tensor)

    @property
    def share(self):
        """Returns underlying share"""
        return self._tensor.share

    @share.setter
    def share(self, value):
        """Sets share to value"""
        self._tensor.share = value

    @property
    def rep_share(self):
        return self._tensor.rep_share

    @rep_share.setter
    def rep_share(self, value):
        self._tensor.rep_share = value

    @property
    def encoder(self):
        """Returns underlying encoder"""
        return self._tensor.encoder

    @encoder.setter
    def encoder(self, value):
        """Sets encoder to value"""
        self._tensor.encoder = value

    @staticmethod
    def __cat_stack_helper(op, tensors, *args, **kwargs):
        assert op in ["cat", "stack"], "Unsupported op for helper function"
        assert isinstance(tensors, list), "%s input must be a list" % op
        assert len(tensors) > 0, "expected a non-empty list of MPCTensors"

        _ptype = kwargs.pop("ptype", None)
        # Populate ptype field
        if _ptype is None:
            for tensor in tensors:
                if isinstance(tensor, MPCTensor):
                    _ptype = tensor.ptype
                    break
        if _ptype is None:
            _ptype = Ptype.arithmetic

        # Make all inputs MPCTensors of given ptype
        for i, tensor in enumerate(tensors):
            if tensor.ptype != _ptype:
                tensors[i] = tensor.to(_ptype)

        # Operate on all input tensors
        result = tensors[0].clone()
        funcs = {"cat": torch_cat, "stack": torch_stack}
        result.share = funcs[op]([tensor.share for tensor in tensors], *args, **kwargs)
        return result

    @staticmethod
    def cat(tensors, *args, **kwargs):
        """Perform matrix concatenation"""
        return MPCTensor.__cat_stack_helper("cat", tensors, *args, **kwargs)

    @staticmethod
    def stack(tensors, *args, **kwargs):
        """Perform tensor stacking"""
        return MPCTensor.__cat_stack_helper("stack", tensors, *args, **kwargs)

    # Comparators
    @mode(Ptype.binary)
    def _ltz(self, _scale=True):
        """Returns 1 for elements that are < 0 and 0 otherwise"""
        shift = torch.iinfo(torch.long).bits - 1
        #result = (self >> shift).to(Ptype.arithmetic, bits=1)
        result = self.to(Ptype.arithmetic, bits=1)
        if _scale:
            return result * result.encoder._scale
        else:
            result.encoder._scale = 1
            return result

    @mode(Ptype.binary)
    def _gez(self):
        return self ^ 1

    @mode(Ptype.arithmetic)
    def _ge(self, y):
        return (self - y)._gez()

    @mode(Ptype.arithmetic)
    def ge(self, y, _scale=True):
        """Returns self >= y"""
        return 1 - self.lt(y, _scale=_scale)

    @mode(Ptype.arithmetic)
    def gt(self, y, _scale=True):
        """Returns self > y"""
        return (-self + y)._ltz(_scale=_scale)

    @mode(Ptype.arithmetic)
    def le(self, y, _scale=True):
        """Returns self <= y"""
        return 1 - self.gt(y, _scale=_scale)

    @mode(Ptype.arithmetic)
    def lt(self, y, _scale=True):
        """Returns self < y"""
        return (self - y)._ltz(_scale=_scale)

    @mode(Ptype.arithmetic)
    def eq(self, y, _scale=True):
        """Returns self == y"""
        if comm.get().get_world_size() == 2:
            return (self - y)._eqz_2PC(_scale=_scale)

        return 1 - self.ne(y, _scale=_scale)

    @mode(Ptype.arithmetic)
    def ne(self, y, _scale=True):
        """Returns self != y"""
        difference = self - y
        difference.share = torch_stack([difference.share, -(difference.share)])
        return difference._ltz(_scale=_scale).sum(0)

    @mode(Ptype.arithmetic)
    def relu(self):
        """Compute a Rectified Linear function on the input tensor."""
        assert comm.get_world_size() == 3
        return MPCTensor.from_shares(
                resharing.mixed_mul(self._tensor, get_msb(self._tensor)^1), 
                src=comm.get().get_rank()
        )

    @mode(Ptype.arithmetic)
    def max(self, dim=None, keepdim=False, one_hot=True):
        """Returns the maximum value of all elements in the input tensor."""
        method = config.max_method
        method = 'log_reduction'
        max_result = _max_helper_all_tree_reductions(self, dim=dim, method=method, keepdim=keepdim)
        return max_result

    @mode(Ptype.arithmetic)
    def min(self, dim=None, keepdim=False, one_hot=True):
        """Returns the minimum value of all elements in the input tensor."""
        # TODO: Make dim an arg.
        result = (-self).max(dim=dim, keepdim=keepdim, one_hot=one_hot)
        if dim is None:
            return -result
        else:
            return -result[0], result[1]

    def where(self, condition, y):
        """Selects elements from self or y based on condition

        Args:
            condition (torch.bool or MPCTensor): when True yield self,
                otherwise yield y
            y (torch.tensor or MPCTensor): values selected at indices
                where condition is False.

        Returns: MPCTensor or torch.tensor
        """
        if is_tensor(condition):
            condition = condition.float()
            y_masked = y * (1 - condition)
        else:
            # encrypted tensor must be first operand
            y_masked = (1 - condition) * y

        return self * condition + y_masked

    def pairwise_max(self, y):
        comp_bit = self._ge(y)
        #pairwise_max = MPCTensor.from_shares(resharing.mixed_mul((self-y)._tensor,  comp_bit._tensor), src=comm.get().get_rank()) + y
        pairwise_max = resharing.mixed_mul((self-y)._tensor,  comp_bit._tensor) + y
        return pairwise_max

    @mode(Ptype.arithmetic)
    def softmax(self, dim, **kwargs):
        """Compute the softmax of a tensor's elements along a given dimension
        """
        # 0-d case
        if self.dim() == 0:
            assert dim == 0, "Improper dim argument"
            return MPCTensor(torch.ones_like((self.share)))

        if self.size(dim) == 1:
            return MPCTensor(torch.ones_like(self.share))

        maximum_value = self.max(dim, keepdim=True)
        logits = self - maximum_value

        numerator = logits.exp()
        with ConfigManager("reciprocal_all_pos", True):
            inv_denominator = numerator.sum(dim, keepdim=True).reciprocal()
        return numerator * inv_denominator
    

    @mode(Ptype.arithmetic)
    def log_softmax(self, dim, **kwargs):
        """Applies a softmax followed by a logarithm.
        While mathematically equivalent to log(softmax(x)), doing these two
        operations separately is slower, and numerically unstable. This function
        uses an alternative formulation to compute the output and gradient correctly.
        """
        # 0-d case
        if self.dim() == 0:
            assert dim == 0, "Improper dim argument"
            return MPCTensor(torch.zeros((), device=self.device))

        if self.size(dim) == 1:
            return MPCTensor(torch.zeros_like(self.share))

        maximum_value = self.max(dim, keepdim=True)[0]
        logits = self - maximum_value
        normalize_term = logits.exp().sum(dim, keepdim=True)
        result = logits - normalize_term.log()
        return result

    @mode(Ptype.arithmetic)
    def pad(self, pad, mode="constant", value=0):
        result = self.shallow_copy()
        if isinstance(value, MPCTensor):
            result._tensor = self._tensor.pad(pad, mode=mode, value=value._tensor)
        else:
            result._tensor = self._tensor.pad(pad, mode=mode, value=value)
        return result

    # Approximations:
    def exp(self):
        """Approximates the exponential function using a limit approximation:

        .. math::

            exp(x) = \lim_{n \\rightarrow \\infty} (1 + x / n) ^ n

        Here we compute exp by choosing n = 2 ** d for some large d equal to
        `iterations`. We then compute (1 + x / n) once and square `d` times.

        Set the number of iterations for the limit approximation with
        config.exp_iterations.
        """  # noqa: W605
        result = 1 + self.div(2 ** config.exp_iterations)
        for _ in range(config.exp_iterations):
            result = result.square()
        
        result_pt = result.get_plain_text()
        return result

    def log(self):
        r"""
        Approximates the natural logarithm using 8th order modified
        Householder iterations. This approximation is accurate within 2% relative
        error on [0.0001, 250].

        Iterations are computed by: :math:`h = 1 - x * exp(-y_n)`

        .. math::

            y_{n+1} = y_n - \sum_k^{order}\frac{h^k}{k}

        Args:
            iterations (int): number of Householder iterations for the approximation
            exp_iterations (int): number of iterations for limit approximation of exp
            order (int): number of polynomial terms used (order of Householder approx)
        """

        # Initialization to a decent estimate (found by qualitative inspection):
        #                ln(x) = x/120 - 20exp(-2x - 1.0) + 3.0
        iterations = config.log_iterations
        exp_iterations = config.log_exp_iterations
        order = config.log_order

        term1 = self.div(120)
        term2 = self.mul(2).add(1.0).neg().exp().mul(20)
        y = term1 - term2 + 3.0

        # 8th order Householder iterations
        with ConfigManager("exp_iterations", exp_iterations):
            for _ in range(iterations):
                h = 1 - self * (-y).exp()
                y -= h.polynomial([1 / (i + 1) for i in range(order)])
        return y


    @mode(Ptype.arithmetic)
    def polynomial(self, coeffs, func="mul"):
        """Computes a polynomial function on a tensor with given coefficients,
        `coeffs`, that can be a list of values or a 1-D tensor.
        Coefficients should be ordered from the order 1 (linear) term first,
        ending with the highest order term. (Constant is not included).
        """
        # Coefficient input type-checking
        if isinstance(coeffs, list):
            coeffs = torch.tensor(coeffs, device=self.device)
        assert is_tensor(coeffs) or crypten.is_encrypted_tensor(
            coeffs
        ), "Polynomial coefficients must be a list or tensor"
        assert coeffs.dim() == 1, "Polynomial coefficients must be a 1-D tensor"

        # Handle linear case
        if coeffs.size(0) == 1:
            return self.mul(coeffs)

        # Compute terms of polynomial using exponentially growing tree
        terms = crypten.stack([self, self.square()])
        while terms.size(0) < coeffs.size(0):
            highest_term = terms.index_select(
                0, torch.tensor(terms.size(0) - 1, device=self.device)
            )
            new_terms = getattr(terms, func)(highest_term)
            terms = crypten.cat([terms, new_terms])

        # Resize the coefficients for broadcast
        terms = terms[: coeffs.size(0)]
        for _ in range(terms.dim() - 1):
            coeffs = coeffs.unsqueeze(1)

        # Multiply terms by coefficients and sum
        return terms.mul(coeffs).sum(0)

    def reciprocal(self):
        """
        Methods:
            'NR' : `Newton-Raphson`_ method computes the reciprocal using iterations
                    of :math:`x_{i+1} = (2x_i - self * x_i^2)` and uses
                    :math:`3*exp(-(x-.5)) + 0.003` as an initial guess by default

            'log' : Computes the reciprocal of the input from the observation that:
                    :math:`x^{-1} = exp(-log(x))`

        Configuration params:
            reciprocal_method (str):  One of 'NR' or 'log'.
            reciprocal_nr_iters (int):  determines the number of Newton-Raphson iterations to run
                         for the `NR` method
            reciprocal_log_iters (int): determines the number of Householder
                iterations to run when computing logarithms for the `log` method
            reciprocal_all_pos (bool): determines whether all elements of the
                input are known to be positive, which optimizes the step of
                computing the sign of the input.
            reciprocal_initial (tensor): sets the initial value for the
                Newton-Raphson method. By default, this will be set to :math:
                `3*exp(-(x-.5)) + 0.003` as this allows the method to converge over
                a fairly large domain

        .. _Newton-Raphson:
            https://en.wikipedia.org/wiki/Newton%27s_method
        """
        method = config.reciprocal_method
        if not config.reciprocal_all_pos:
            sgn = self.sign(_scale=False)
            pos = sgn * self
            with ConfigManager("reciprocal_all_pos", True):
                return sgn * pos.reciprocal()

        if method == "NR":
            if config.reciprocal_initial is None:
                # Initialization to a decent estimate (found by qualitative inspection):
                #                1/x = 3exp(.5 - x) + 0.003
                result = 3 * (0.5 - self).exp() + 0.003
            else:
                result = config.reciprocal_initial
            for _ in range(config.reciprocal_nr_iters):
                if isinstance(result, MPCTensor):
                    result += result - result.square().mul_(self)
                else:
                    result = 2 * result - result * result * self
                
            return result
        else:
            raise ValueError(f"Invalid method {method} given for reciprocal function")

    def div(self, y):
        r"""Divides each element of :attr:`self` with the scalar :attr:`y` or
        each element of the tensor :attr:`y` and returns a new resulting tensor.

        For `y` a scalar:

        .. math::
            \text{out}_i = \frac{\text{self}_i}{\text{y}}

        For `y` a tensor:

        .. math::
            \text{out}_i = \frac{\text{self}_i}{\text{y}_i}

        Note for :attr:`y` a tensor, the shapes of :attr:`self` and :attr:`y` must be
        `broadcastable`_.

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics"""  # noqa: B950
        result = self.clone()
        if isinstance(y, CrypTensor):
            result.share = torch.broadcast_tensors(result.share, y.share)[0].clone()
        elif is_tensor(y):
            result.share = torch.broadcast_tensors(result.share, y)[0].clone()
        return result.div_(y)

    def div_(self, y):
        """In-place version of :meth:`div`"""
        if isinstance(y, MPCTensor):
            return self.mul_(y.reciprocal())
        self._tensor.div_(y)
        return self

    def pow(self, p, **kwargs):
        """
        Computes an element-wise exponent `p` of a tensor, where `p` is an
        integer.
        """
        if isinstance(p, float) and int(p) == p:
            p = int(p)

        if not isinstance(p, int):
            raise TypeError(
                "pow must take an integer exponent. For non-integer powers, use"
                " pos_pow with positive-valued base."
            )
        if p < -1:
            return self.reciprocal().pow(-p)
        elif p == -1:
            return self.reciprocal()
        elif p == 0:
            # Note: This returns 0 ** 0 -> 1 when inputs have zeros.
            # This is consistent with PyTorch's pow function.
            return MPCTensor(torch.ones_like(self.share))
        elif p == 1:
            return self.clone()
        elif p == 2:
            return self.square()
        elif p % 2 == 0:
            return self.square().pow(p // 2)
        else:
            return self.square().mul_(self).pow((p - 1) // 2)

    def pow_(self, p, **kwargs):
        """In-place version of pow_ function"""
        result = self.pow(p)
        self.share.set_(result.share.data)
        return self

    def pos_pow(self, p):
        """
        Approximates self ** p by computing: :math:`x^p = exp(p * log(x))`

        Note that this requires that the base `self` contain only positive values
        since log can only be computed on positive numbers.

        Note that the value of `p` can be an integer, float, public tensor, or
        encrypted tensor.
        """
        if isinstance(p, int) or (isinstance(p, float) and int(p) == p):
            return self.pow(p)
        return self.log().mul_(p).exp()

    def sqrt(self):
        """
        Computes the square root of the input by raising it to the 0.5 power
        """
        return self.pos_pow(0.5)

    def unbind(self, dim=0):
        shares = self.share.unbind(dim=dim)
        results = tuple(
            MPCTensor(0, ptype=self.ptype, device=self.device)
            for _ in range(len(shares))
        )
        for i in range(len(shares)):
            results[i].share = shares[i]
        return results

    def split(self, split_size, dim=0):
        shares = self.share.split(split_size, dim=dim)
        results = tuple(
            MPCTensor(0, ptype=self.ptype, device=self.device)
            for _ in range(len(shares))
        )
        for i in range(len(shares)):
            results[i].share = shares[i]
        return results

    def set(self, enc_tensor):
        """
        Sets self encrypted to enc_tensor in place by setting
        shares of self to those of enc_tensor.

        Args:
            enc_tensor (MPCTensor): with encrypted shares.
        """
        if is_tensor(enc_tensor):
            enc_tensor = MPCTensor(enc_tensor)
        assert isinstance(enc_tensor, MPCTensor), "enc_tensor must be an MPCTensor"
        self.share.set_(enc_tensor.share)
        return self


OOP_UNARY_FUNCTIONS = {
    "avg_pool2d": Ptype.arithmetic,
    "sum_pool2d": Ptype.arithmetic,
    "take": Ptype.arithmetic,
    "square": Ptype.arithmetic,
    "mean": Ptype.arithmetic,
    "var": Ptype.arithmetic,
    "neg": Ptype.arithmetic,
    "__neg__": Ptype.arithmetic,
    "invert": Ptype.binary,
    "lshift": Ptype.binary,
    "rshift": Ptype.binary,
    "__invert__": Ptype.binary,
    "__lshift__": Ptype.binary,
    "__rshift__": Ptype.binary,
    "__rand__": Ptype.binary,
    "__rxor__": Ptype.binary,
    "__ror__": Ptype.binary,
}

OOP_BINARY_FUNCTIONS = {
    "add": Ptype.arithmetic,
    "sub": Ptype.arithmetic,
    "mul": Ptype.arithmetic,
    "matmul": Ptype.arithmetic,
    "conv1d": Ptype.arithmetic,
    "conv2d": Ptype.arithmetic,
    "conv_transpose1d": Ptype.arithmetic,
    "conv_transpose2d": Ptype.arithmetic,
    "dot": Ptype.arithmetic,
    "ger": Ptype.arithmetic,
    "__xor__": Ptype.binary,
    "__or__": Ptype.binary,
    "__and__": Ptype.binary,
}

INPLACE_UNARY_FUNCTIONS = {
    "neg_": Ptype.arithmetic,
    "invert_": Ptype.binary,
    "lshift_": Ptype.binary,
    "rshift_": Ptype.binary,
}

INPLACE_BINARY_FUNCTIONS = {
    "add_": Ptype.arithmetic,
    "sub_": Ptype.arithmetic,
    "mul_": Ptype.arithmetic,
    "__ior__": Ptype.binary,
    "__ixor__": Ptype.binary,
    "__iand__": Ptype.binary,
}


def _add_oop_unary_passthrough_function(name, preferred=None):
    def ou_wrapper_function(self, *args, **kwargs):
        result = self.shallow_copy()
        result._tensor = getattr(result._tensor, name)(*args, **kwargs)
        return result

    if preferred is None:
        setattr(MPCTensor, name, ou_wrapper_function)
    else:
        setattr(MPCTensor, name, mode(preferred, False)(ou_wrapper_function))


def _add_oop_binary_passthrough_function(name, preferred=None):
    def ob_wrapper_function(self, value, *args, **kwargs):
        result = self.shallow_copy()
        if isinstance(value, MPCTensor):
            value = value._tensor
        result._tensor = getattr(result._tensor, name)(value, *args, **kwargs)
        return result

    if preferred is None:
        setattr(MPCTensor, name, ob_wrapper_function)
    else:
        setattr(MPCTensor, name, mode(preferred, False)(ob_wrapper_function))


def _add_inplace_unary_passthrough_function(name, preferred=None):
    def iu_wrapper_function(self, *args, **kwargs):
        self._tensor = getattr(self._tensor, name)(*args, **kwargs)
        return self

    if preferred is None:
        setattr(MPCTensor, name, iu_wrapper_function)
    else:
        setattr(MPCTensor, name, mode(preferred, True)(iu_wrapper_function))


def _add_inplace_binary_passthrough_function(name, preferred=None):
    def ib_wrapper_function(self, value, *args, **kwargs):
        if isinstance(value, MPCTensor):
            value = value._tensor
        self._tensor = getattr(self._tensor, name)(value, *args, **kwargs)
        return self

    if preferred is None:
        setattr(MPCTensor, name, ib_wrapper_function)
    else:
        setattr(MPCTensor, name, mode(preferred, True)(ib_wrapper_function))


for func_name, preferred_type in OOP_UNARY_FUNCTIONS.items():
    _add_oop_unary_passthrough_function(func_name, preferred_type)

for func_name, preferred_type in OOP_BINARY_FUNCTIONS.items():
    _add_oop_binary_passthrough_function(func_name, preferred_type)

for func_name, preferred_type in INPLACE_UNARY_FUNCTIONS.items():
    _add_inplace_unary_passthrough_function(func_name, preferred_type)

for func_name, preferred_type in INPLACE_BINARY_FUNCTIONS.items():
    _add_inplace_binary_passthrough_function(func_name, preferred_type)


REGULAR_FUNCTIONS = [
    "__getitem__",
    "index_select",
    "view",
    "flatten",
    "t",
    "transpose",
    "unsqueeze",
    "squeeze",
    "repeat",
    "narrow",
    "expand",
    "roll",
    "unfold",
    "flip",
    "trace",
    "prod",
    "sum",
    "cumsum",
    "reshape",
    "gather",
    "permute",
]


PROPERTY_FUNCTIONS = ["__len__", "nelement", "dim", "size", "numel"]


def _add_regular_function(function_name):
    """
    Adds function to `MPCTensor` that is applied directly on the underlying
    `_tensor` attribute, and stores the result in the same attribute.
    """

    def regular_func(self, *args, **kwargs):
        result = self.shallow_copy()
        result._tensor = getattr(result._tensor, function_name)(*args, **kwargs)
        return result

    setattr(MPCTensor, function_name, regular_func)


def _add_property_function(function_name):
    """
    Adds function to `MPCTensor` that is applied directly on the underlying
    `_tensor` attribute, and returns the result of that function.
    """

    def property_func(self, *args, **kwargs):
        return getattr(self._tensor, function_name)(*args, **kwargs)

    setattr(MPCTensor, function_name, property_func)


for function_name in REGULAR_FUNCTIONS:
    _add_regular_function(function_name)

for function_name in PROPERTY_FUNCTIONS:
    _add_property_function(function_name)
