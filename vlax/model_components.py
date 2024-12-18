import logging
import os
from functools import partial

import cloudpickle
import jax
import numpy as np
from flax.core import freeze
from jax.experimental import multihost_utils as mhu
from transformers import AutoTokenizer

from vlax.bc.eval_step import compute_metrics
from vlax.bc.train_step import VLABatch, train_step
from vlax.models.big_vision.predict_fns import get_all
from vlax.models.optimizer import make_optimizer
from vlax.models.vla import Model as VLAModel
from vlax.sequence.action_tokenizer import ActionTokenizer
from vlax.sequence.sequence import SequenceWithSegments
from vlax.sharding import ShardingConfig
from vlax.train_state import TrainState


def _make_model(model_config):
    return VLAModel(**freeze(model_config))


def _make_optimizer(optimizer_config):
    return make_optimizer(**freeze(optimizer_config))


def make_train_state(
    *,
    model_config: dict,
    optimizer_config: dict,
    example_batch,
    sharding: ShardingConfig,
    seed: int = 0,
):
    model = _make_model(model_config)
    optimizer = _make_optimizer(optimizer_config)

    def init_train_state(example_batch):
        rng, key = jax.random.split(jax.random.PRNGKey(seed))
        params = model.lazy_init(key, example_batch, method="init_helper")["params"]
        return TrainState.create(
            apply_fn=model.apply,
            model=model,
            params=params,
            tx=optimizer,
            rng_key=rng,
        )

    init_train_state = sharding.mesh.sjit(
        init_train_state,
        in_shardings=None,
        out_shardings=sharding.model_sharding,
    )

    return init_train_state(example_batch)


class ModelComponents:
    def __init__(
        self,
        model_config: dict,
        optimizer_config: dict,
        sharding: ShardingConfig,
        language_tokenizer: AutoTokenizer,
        action_tokenizer: ActionTokenizer,
        example_batch,
        rng_key: np.random.Generator,
        max_loss_tokens: int | None = None,
    ):
        self.model_train_state = make_train_state(
            model_config=model_config,
            optimizer_config=optimizer_config,
            example_batch=example_batch,
            sharding=sharding,
            seed=rng_key.integers(0, 2**32),
        )
        self.max_loss_tokens = max_loss_tokens
        self.sharding = sharding
        self.language_tokenizer: AutoTokenizer = language_tokenizer
        self.action_tokenizer = action_tokenizer
        self.example_batch = example_batch
        self.rng_key = rng_key
        self._train_step = sharding.mesh.sjit(
            partial(train_step, max_loss_tokens=max_loss_tokens),
            in_shardings=(sharding.model_sharding, sharding.data_sharding),
            out_shardings=(sharding.model_sharding, None),
            args_sharding_constraint=(sharding.model_sharding, sharding.data_sharding),
        )
        self._predict_fns = get_all(self.model_train_state.model)

    def save_non_state_components(self, path):
        with open(os.path.join(path, "model.pkl"), "wb") as f:
            cloudpickle.dump(self.model_train_state.model, f)
        with open(os.path.join(path, "optimizer.pkl"), "wb") as f:
            cloudpickle.dump(self.model_train_state.tx, f)
        with open(os.path.join(path, "example_batch.pkl"), "wb") as f:
            cloudpickle.dump(self.example_batch, f)
        self.language_tokenizer.save_pretrained(
            os.path.join(path, "language_tokenizer")
        )
        self.action_tokenizer.save(os.path.join(path, "action_tokenizer"))

    @classmethod
    def restore_non_state_components(cls, path, sharding: ShardingConfig):
        model = cloudpickle.load(os.path.join(path, "model.pkl"))
        tx = cloudpickle.load(os.path.join(path, "optimizer.pkl"))
        example_batch = cloudpickle.load(os.path.join(path, "example_batch.pkl"))
        language_tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(path, "language_tokenizer")
        )

        action_tokenizer = ActionTokenizer.load(os.path.join(path, "action_tokenizer"))
        return cls(
            make_train_state(model, tx, example_batch, sharding),
            language_tokenizer,
            action_tokenizer,
            example_batch,
        )

    def train_step(self, batch: VLABatch):
        if self.max_loss_tokens is not None:
            actual_loss_tokens = np.sum(batch.sequence.mask_loss, axis=-1)
            actual_loss_tokens = mhu.process_allgather(actual_loss_tokens).max()
            if actual_loss_tokens > self.max_loss_tokens:
                logging.warning(
                    f"Max loss tokens {actual_loss_tokens} is greater than {self.max_loss_tokens}"
                )

        batch = self.sharding.mesh.local_data_to_global_array(
            batch,
            batch_axis=0,
        )
        self.model_train_state, info = self._train_step(
            self.model_train_state,
            batch,
        )
        return info

    def eval_step(self, batch):
        model_inputs = self.sharding.mesh.local_data_to_global_array(
            {
                "sequence": batch["data"]["prompt_tokens"],
                "data": batch["data"]["observation"],
                "data_mask": batch["data_mask"]["observation"],
            },
            batch_axis=0,
        )
        prediction_tokens = self._predict_fns["decode"](
            self.model_train_state, **model_inputs
        )
        predictions = self.action_tokenizer.decode(prediction_tokens)
        return compute_metrics(
            batch | {"pred_tokens": prediction_tokens, "pred_actions": predictions}
        )

    def predict_action_batched(
        self, sequence: SequenceWithSegments, data: jax.Array, data_mask: jax.Array
    ):
        tokens = self._predict_fns["decode"](
            self.model_train_state, sequence, data, data_mask
        )
        return self.action_tokenizer.decode(tokens)
