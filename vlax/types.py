import jax
from chex import dataclass
from jax import Array

from vlax.sequence.sequence import SequenceWithSegments


@dataclass
class VLABatch:
    sequence: SequenceWithSegments
    data: Array
    data_mask: jax.Array
