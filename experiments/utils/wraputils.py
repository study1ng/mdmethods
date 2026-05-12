from functools import wraps
from fractions import Fraction
from typing import Callable
from experiments.assertions import AssertEq, Assert


def repeat(
    v, dim: int, *, types: type | tuple[type, ...] | None = None, wrap_type=tuple
):
    """if v is types, then repeat it for dim times

    Parameters
    ----------
    v : any
        any value to be repeated
    dim : int
        repeat times
    types : type | tuple[type, ...] | None
        target types if None then always wrap
    wrap_type : Function which get a Iterable, optional
        the wrapper, by default tuple

    Returns
    -------
    wrap_type
        the wrapped value

    Notes
    -----
    This function don't assert len(returns) == dim, if v is not instance of types
    """
    if types is None:
        types = type(v)
    if not isinstance(v, types):
        return v
    return wrap_type(v for _ in range(dim))


def element_wise(types=(int, float, Fraction), wrap_type=tuple):
    """wrapper to make a function can be used for Iterable

    Parameters
    ----------
    types : type | tuple[type, ...]
        the target type
    wrap_type : function which get a Iterable, optional
        the wrapper of return, by default tuple

    Examples
    --------
    ```python
    @element_wise(int)
    def pr(a):
        print(a)
    pr(1)
    >>> 1
    pr((1, 2))
    >>> 1
    >>> 2
    pr("abc") # due to str is not int, str is handled as a iterable and each character enter pr
    >>> a
    >>> b
    >>> c
    ```
    """

    def _element_wise(func):
        @wraps(func)
        def wrapper(arg):
            if isinstance(arg, types):
                return func(arg)
            else:
                return wrap_type(func(a) for a in arg)

        return wrapper

    return _element_wise


def element_wise2(types=(int, float, Fraction), wrap_type=tuple):
    """like element_wise, but get two argument. if both is not types, assert its length is same

    Parameters
    ----------
    types : type | tuple[type, ...]
        the target type
    wrap_type : function which get a Iterable, optional
        the wrapper of return, by default tuple

    Examples
    --------
    ```python
    @element_wise2(int)
    def add(a, b):
        return a + b
    assert add(1, 2) == 3
    assert add((1, 2), 3) == (4, 5)
    assert add(3, (1, 2)) == (4, 5)
    assert add((1, 2), (3, 4)) == (4, 6)
    assert add("ab", "de") == ("ad", "be") # str is not int so it will be handled as a Iterable and each element character enter add
    """

    def _element_wise2(func):
        @wraps(func)
        def wrapper(arg1, arg2):
            if isinstance(arg1, types) and isinstance(arg2, types):
                return func(arg1, arg2)
            if isinstance(arg1, types) and not isinstance(arg2, types):
                return wrap_type(func(arg1, a) for a in arg2)
            if not isinstance(arg1, types) and isinstance(arg2, types):
                return wrap_type(func(a, arg2) for a in arg1)
            return wrap_type(func(a, b) for a, b in zip(arg1, arg2, strict=True))

        return wrapper

    return _element_wise2


@element_wise2()
def elementwise_div(l, r):
    return l / r


@element_wise2()
def elementwise_is_divisor(l, r):
    return l % r == 0


@element_wise2()
def elementwise_intdiv(l, r):
    return l // r


@element_wise2()
def elementwise_mul(l, r):
    return l * r


@element_wise2()
def elementwise_gt(l, r):
    return l > r


@element_wise2()
def elementwise_le(l, r):
    return l <= r


@element_wise2()
def elementwise_min(l, r):
    return min(l, r)


@element_wise2()
def elementwise_max(l, r):
    return max(l, r)


def divmod_accept_tuple(
    lhs: int | tuple[int, ...], rhs: int | tuple[int, ...]
) -> tuple[int | tuple[int, ...], int | tuple[int, ...]]:
    """divmod(lhs, rhs), but accept tuple

    Parameters
    ----------
    lhs : int | tuple[int, ...]

    rhs : int | tuple[int, ...]

    Returns
    -------
    tuple[Union[int, tuple[int, ...]], Union[int, tuple[int, ...]]]
        (div, mod)
    """
    return (elementwise_intdiv(lhs, rhs), elementwise_is_divisor(lhs, rhs))


def assert_divisible(
    lhs: int | tuple[int, ...], rhs: int | tuple[int, ...]
) -> int | tuple[int, ...]:
    """assert rhs is all divisors of lhs, and return lhs // rhs

    Parameters
    ----------
    lhs : int | tuple[int, ...]

    rhs : int | tuple[int, ...]


    Returns
    -------
    Union[int, tuple[int, ...]]
        lhs // rhs
    """
    div, m = divmod_accept_tuple(lhs, rhs)
    Assert(msg=f"not dividable, lhs: {lhs}, rhs: {rhs}")(all(repeat(m, 1, types=bool)))
    return div


def identity(x):
    return x


@element_wise()
def numerator(i: Fraction):
    return to_fraction(i).numerator


@element_wise()
def denominator(i: Fraction):
    return to_fraction(i).denominator


@element_wise()
def assert_to_integer(i: Fraction):
    i = to_fraction(i)
    AssertEq()(1, i.denominator)
    return i.numerator


@element_wise2()
def elementwise_eq(i, j):
    return i == j


def is_integer(i):
    return elementwise_eq(denominator(i), 1)


@element_wise2()
def assert_integer_scale(shape: int, scale: Fraction):
    out = shape * scale
    return assert_to_integer(out)


@element_wise()
def to_fraction(f):
    return Fraction(f)


@element_wise2()
def to_fraction_with_denominator(n, d):
    return Fraction(n, d)


def scale_shape_fn(scale) -> Callable[[int | tuple[int, ...]], int | tuple[int, ...]]:

    scale = to_fraction(scale)

    def wrapper(shape: int):
        return assert_to_integer(elementwise_mul(shape, scale))

    return wrapper


@element_wise()
def reciprocal(i):
    return Fraction(1, i)
