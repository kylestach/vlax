import jax
import numpy as np
from jax.experimental import multihost_utils


def key_string(path, separator="/") -> str:
    def _component_to_string(component) -> str:
        if isinstance(component, jax.tree_util.SequenceKey):
            return str(component.idx)
        elif isinstance(component, jax.tree_util.DictKey):
            return str(component.key)
        elif isinstance(component, jax.tree_util.GetAttrKey):
            return str(component.name)
        elif isinstance(component, jax.tree_util.FlattenedIndexKey):
            return str(component.key)
        else:
            return str(component)

    return separator.join(_component_to_string(component) for component in path)


def flatten_dict(d):
    leaves = jax.tree_util.tree_leaves_with_path(d)
    return {key_string(path): value for path, value in leaves}


def host_broadcast_str(x: str | None) -> str:
    """
    Broadcast_one_to_all, but with a string.

    Works by padding the string to the length of the longest string and then
    broadcasting the result, then stripping the padding.

    Note: this will remove the padding from the end of the string.
    """
    if x is None:
        x = ""

    max_len = multihost_utils.broadcast_one_to_all(len(x))
    padded = x.ljust(max_len)

    encoded = np.array([ord(c) for c in padded], dtype=np.uint8)[:max_len]
    encoded = multihost_utils.broadcast_one_to_all(encoded)
    decoded = "".join([chr(u) for u in encoded])

    return decoded.rstrip()
