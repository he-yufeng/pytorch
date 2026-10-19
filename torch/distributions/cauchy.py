# mypy: allow-untyped-defs
import math

import torch
from torch import inf, nan, Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all
from torch.types import _Number, _size


__all__ = ["Cauchy"]


class Cauchy(Distribution):
    r"""
    Samples from a Cauchy (Lorentz) distribution. The distribution of the ratio of
    independent normally distributed random variables with means `0` follows a
    Cauchy distribution.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Cauchy distribution with loc=0 and scale=1
        tensor([ 2.3214])

    Args:
        loc (float or Tensor): mode or median of the distribution.
        scale (float or Tensor): half width at half maximum.
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, _Number) and isinstance(scale, _Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Cauchy, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Cauchy, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> Tensor:
        return torch.full(
            self._extended_shape(), nan, dtype=self.loc.dtype, device=self.loc.device
        )

    @property
    def mode(self) -> Tensor:
        return self.loc

    @property
    def variance(self) -> Tensor:
        return torch.full(
            self._extended_shape(), inf, dtype=self.loc.dtype, device=self.loc.device
        )

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        eps = self.loc.new(shape).cauchy_()
        return self.loc + eps * self.scale

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return (
            -math.log(math.pi)
            - self.scale.log()
            - (((value - self.loc) / self.scale) ** 2).log1p()
        )

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # atan(z)/pi + 0.5 annihilates the lower tail (z -> -inf rounds to
        # exactly 0.5), while the atan2 form below is exact there but loses
        # the upper tail to 1 - eps rounding. Evaluate each on its own side.
        lower = torch.atan2(self.scale, self.loc - value) / math.pi
        upper = 1 - torch.atan2(self.scale, value - self.loc) / math.pi
        return torch.where(value <= self.loc, lower, upper)

    def icdf(self, value):
        # tan(pi*(q-0.5)) drops the low digits of q in the subtraction, and
        # float pi puts the q=1 pole at a large finite value. The cotangent
        # form keeps the tail digits on both sides: q directly below 0.5,
        # 1-q (exact by Sterbenz) above it.
        lower = -self.scale / torch.tan(math.pi * value)
        upper = self.scale / torch.tan(math.pi * (1 - value))
        return torch.where(value <= 0.5, lower, upper) + self.loc

    def entropy(self):
        return math.log(4 * math.pi) + self.scale.log()
