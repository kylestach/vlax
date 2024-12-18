import jax
import jax.numpy as jnp
import optax

from vlax.sequence.sequence import SequenceWithSegments
from vlax.train_state import TrainState
from vlax.types import VLABatch


def train_step(model: TrainState, batch: VLABatch, max_loss_tokens: int | None):
    rng = model.rng_key

    def loss_fn(params, batch: VLABatch, rng):
        overall_sequence: SequenceWithSegments = batch.sequence
        input_sequence: SequenceWithSegments = batch.sequence[:-1]
        target_sequence: SequenceWithSegments = batch.sequence[1:]

        all_pre_logits, segment_pre_logits, _ = model.model.apply(
            {"params": params},
            sequence=input_sequence,
            data=batch.data,
            data_mask=batch.data_mask,
            train=True,
            pre_logits=True,
            rngs={"dropout": rng},
        )

        target_tokens = target_sequence.data
        target_loss_mask = target_sequence.mask_loss

        if max_loss_tokens is not None:
            indices = jnp.argsort(target_loss_mask, axis=1)[:, -max_loss_tokens:]
            target_tokens = jnp.take_along_axis(target_tokens, indices, axis=1)
            target_loss_mask = jnp.take_along_axis(target_loss_mask, indices, axis=1)
            all_pre_logits = jnp.take_along_axis(
                all_pre_logits, indices[..., None], axis=1
            )

        def _compute_logits(pre_logits):
            return model.model.apply(
                {"params": params},
                pre_logits=pre_logits,
                train=True,
                method=model.model.compute_logits,
                rngs={"dropout": rng},
            )

        def _masked_stats(pre_logits, target_tokens, target_mask):
            logits = _compute_logits(pre_logits)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits,
                target_tokens,
            )
            loss = (
                jnp.mean(loss * target_mask, axis=-1)
                / (1e-3 + jnp.mean(target_mask, axis=-1))
            ).mean()

            accuracy = jnp.argmax(logits, axis=-1) == target_tokens
            accuracy = jnp.mean(accuracy * target_mask) / (
                1e-3 + jnp.mean(target_mask, axis=-1)
            )

            return loss, accuracy

        all_logits = _compute_logits(all_pre_logits)
        total_loss, total_accuracy = _masked_stats(
            all_logits, target_tokens, target_loss_mask
        )

        # Compute action loss metrics
        action_pre_logits = segment_pre_logits["action"][..., :-1]
        action_logits = _compute_logits(action_pre_logits)
        target_sequence_action = overall_sequence.extract_segments()["action"][1:]
        action_loss, action_accuracy = _masked_stats(
            action_logits, target_sequence_action.data, target_sequence_action.mask_loss
        )

        return total_loss, {
            "tf": {
                "action": {
                    "loss": action_loss,
                    "accuracy": action_accuracy,
                },
                "total": {
                    "loss": total_loss,
                    "accuracy": total_accuracy,
                },
            },
        }

    rng, key = jax.random.split(rng)
    grads, info = jax.grad(loss_fn, has_aux=True)(model.params, batch, key)

    info["optimizer"] = {
        "hyperparams": model.opt_state.hyperparams,
    }

    return model.apply_gradients(grads, rng_key=rng), info
