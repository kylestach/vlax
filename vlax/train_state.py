from typing import TypeVar

import flax.linen as nn
import jax
import optax
from flax.struct import dataclass, field
from flax.training.train_state import TrainState as FlaxTrainState

ModelType = TypeVar("ModelType", bound=nn.Module)


@dataclass
class TrainState(FlaxTrainState):
    model: ModelType = field(pytree_node=False)
    rng_key: jax.Array = field(pytree_node=True)


@dataclass
class EMATrainState(TrainState):
    ema_params: optax.Params
    ema_amount: float

    def apply_gradients(self, grads, **kwargs):
        self = super().apply_gradients(grads, **kwargs)
        return self.replace(
            ema_params=optax.tree_utils.tree_update_moment(
                self.params, self.ema_params, self.ema_amount, 1
            )
        )
