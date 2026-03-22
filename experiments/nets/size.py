from dataclasses import dataclass
from fractions import Fraction

from experiments.utils.wraputils import assert_to_integer

Elemental = int | Fraction

@dataclass(init=False, frozen=True)
class DynamicSize:
    scale: Fraction
    fixed: int

    def __init__(self, scale: Fraction = Fraction(1,), fixed: int = 0):
        self.scale = scale
        self.fixed = fixed

    @classmethod
    def to_dynamicsize(cls, i: Elemental) -> "DynamicSize":
        if isinstance(i, Fraction):
            return cls(scale=i)
        elif isinstance(i, int):
            return cls(margin=i)
        else:
            raise TypeError(f"i is needed to be a int | Fraction but {type(i).__name__}")

    
    def is_scale(self):
        return self.fixed == 0

    
    def is_fixed(self):
        return self.scale == Fraction(1)

    
    def __neg__(self) -> "DynamicSize":
        return DynamicSize(scale=-self.scale, fixed=-self.fixed)
    

    def __pos__(self):
        return DynamicSize(scale=+self.scale, fixed=+self.scale)

    
    def __add__(self, other: Elemental | "DynamicSize") -> "DynamicSize":
        other = DynamicSize.to_dynamicsize(other)
        return DynamicSize(scale=self.scale + other.scale, fixed=self.fixed + other.fixed)
    

    def __sub__(self, other: Elemental | "DynamicSize"):
        other = DynamicSize.to_dynamicsize(other)
        return DynamicSize(scale=self.scale - other.scale, fixed=self.fixed - other.fixed)
    
    def __mul__(self, other: Elemental) -> "DynamicSize":
        fixed = assert_to_integer(other * self.fixed)
        return DynamicSize(scale=self.scale * other, fixed=fixed)
    

    def __div__(self, other: Elemental) -> "DynamicSize":
        fixed = assert_to_integer(self.fixed / Fraction(other))
        return DynamicSize(scale=self.scale / other, fixed=fixed)
    

    def __abs__(self):
        return DynamicSize(scale=abs(self.scale), fixed=abs(self.scale))

