import os
from typing import Protocol

import cloudpickle
import numpy as np
from einops import rearrange

from vlax.models.big_vision.registry import Registry


class ActionTokenizer(Protocol):
    def tokenize(self, data, obs=None): ...

    def detokenize(self, tokens, obs=None, action_dim: int | None = None): ...

    @classmethod
    def create(cls, tokenizer_name: str, *args, **kwargs) -> "ActionTokenizer":
        return Registry.lookup(tokenizer_name, *args, **kwargs)

    @classmethod
    def load(cls, path) -> "ActionTokenizer":
        with open(os.path.join(path, "action_tokenizer.pkl"), "rb") as f:
            return cloudpickle.load(f)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "action_tokenizer.pkl"), "wb") as f:
            cloudpickle.dump(self, f)


@Registry.register("action_tokenizer.bin")
class BinActionTokenizer(ActionTokenizer):
    __slots__ = ["min_action_value", "max_action_value", "action_vocab_size"]

    def __init__(
        self,
        min_action_value: np.ndarray,
        max_action_value: np.ndarray,
        action_vocab_size: int,
    ):
        self.min_action_value = min_action_value
        self.max_action_value = max_action_value
        self.action_vocab_size = action_vocab_size

    @property
    def vocab_size(self):
        return self.action_vocab_size

    def tokenize(self, data, obs=None):
        # Assume normalization and clipping to [-1, 1]
        data = np.clip(data, self.min_action_value, self.max_action_value)
        data = (data - self.min_action_value) / (
            self.max_action_value - self.min_action_value
        )
        data = rearrange(data, "... p a -> ... (p a)")
        return np.clip(
            (data * self.vocab_size).astype(np.int32),
            0,
            self.vocab_size - 1,
        )

    def detokenize(self, tokens, obs=None, action_dim: int | None = None):
        values = tokens / self.vocab_size
        values = np.where((values < -1) | (values > 1), np.nan, values)
        data = (
            values * (self.max_action_value - self.min_action_value)
            + self.min_action_value
        )
        data = rearrange(data, "... (p a) -> ... p a", a=action_dim)
        return data
