from torch import Tensor
from jaxtyping import (
    Float as JaxFloat,
    Int as JaxInt,
    Bool as JaxBool,
    Shaped as JaxShaped,
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

# PyTorch-compatible type annotations
Shaped = TorchTyping(JaxShaped)
Float = TorchTyping(JaxFloat)
Int = TorchTyping(JaxInt)
Bool = TorchTyping(JaxBool)

# Type checking is disabled by default
should_typecheck = False
typecheck = identity
beartype_isinstance = always(True)
__all__ = [
    'Float',
    'Int',
    'Bool',
    'typecheck',
    'beartype_isinstance'
]
