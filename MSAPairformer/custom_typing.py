from functools import wraps
from torch import Tensor
from jaxtyping import (
    Float,
    Int,
    Bool,
    Shaped,
    jaxtyped
)

def always(value):
    def inner(*args, **kwargs):
        return value
    return inner

def identity(t):
    return t

class TorchTyping:
    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        return self.abstract_dtype[Tensor, shapes]

Shaped = TorchTyping(Shaped)
Float = TorchTyping(Float)
Int   = TorchTyping(Int)
Bool  = TorchTyping(Bool)

should_typecheck = False

typecheck = identity

beartype_isinstance = always(True)

__all__ = [
    Float,
    Int,
    Bool,
    typecheck,
    beartype_isinstance
]
