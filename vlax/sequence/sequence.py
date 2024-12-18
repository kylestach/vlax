from typing import Dict

import chex
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass


@dataclass
class Sequence:
    """
    A sequence of tokens (discrete) or embeddings (continuous) with associated masks.
    """

    data: np.ndarray
    mask_exists: np.ndarray
    mask_ar: np.ndarray
    mask_loss: np.ndarray


@dataclass
class SequenceWithSegments(Sequence):
    segments: Dict[str, jnp.ndarray]

    def fill(
        self,
        segment_values: Dict[str, np.ndarray],
        segment_masks: Dict[str, np.ndarray],
    ):
        for segment_name, values in segment_values.items():
            embedding_idcs = self.segments[segment_name]
            embedding_idcs_value = embedding_idcs
            while embedding_idcs_value.ndim < self.data.ndim:
                embedding_idcs_value = embedding_idcs_value[..., None]
            np.put_along_axis(self.data, embedding_idcs_value, values, axis=-1)
            np.put_along_axis(
                self.mask_exists,
                embedding_idcs_value,
                segment_masks[segment_name],
                axis=-1,
            )

    def fill_jax(
        self,
        segment_values: Dict[str, jnp.ndarray],
        segment_masks: Dict[str, jnp.ndarray],
    ):
        batch_shape = self.mask_loss.shape[:-1]
        data_shape = self.data.shape[self.mask_loss.ndim :]

        chex.assert_shape(
            list(segment_values.values()), [*batch_shape, None, *data_shape]
        )
        chex.assert_shape(list(segment_masks.values()), [*batch_shape, None])

        token_sequence = self.data
        mask_exists = self.mask_exists

        for segment_name in segment_values:
            segment_value = segment_values[segment_name]
            segment_mask = segment_masks[segment_name]

            embedding_idcs = self.segments[segment_name]
            embedding_idcs_value = embedding_idcs
            while embedding_idcs_value.ndim < self.data.ndim:
                embedding_idcs_value = embedding_idcs_value[..., None]

            embedding_idcs_value = jnp.broadcast_to(
                embedding_idcs_value, segment_value.shape
            )

            sequence_axis = embedding_idcs.ndim - 1
            token_sequence = jnp.put_along_axis(
                token_sequence,
                embedding_idcs_value,
                segment_value,
                axis=sequence_axis,
                inplace=False,
            )
            segment_mask = segment_mask & jnp.take_along_axis(
                mask_exists,
                embedding_idcs,
                axis=sequence_axis,
            )
            mask_exists = jnp.put_along_axis(
                mask_exists,
                embedding_idcs,
                segment_mask,
                axis=sequence_axis,
                inplace=False,
            )

        return Sequence(
            data=token_sequence,
            mask_exists=mask_exists,
            mask_ar=self.mask_ar,
            mask_loss=self.mask_loss,
        )

    def extract_segments(self) -> Dict[str, Sequence]:
        out = {}
        for segment_name, segment in self.segments.items():
            value_idcs = segment
            while value_idcs.ndim < self.data.ndim:
                value_idcs = value_idcs[..., None]

            sequence_axis = segment.ndim - 1
            out[segment_name] = Sequence(
                data=jnp.take_along_axis(self.data, value_idcs, axis=sequence_axis),
                mask_exists=jnp.take_along_axis(
                    self.mask_exists, segment, axis=sequence_axis
                ),
                mask_ar=jnp.take_along_axis(self.mask_ar, segment, axis=sequence_axis),
                mask_loss=jnp.take_along_axis(
                    self.mask_loss, segment, axis=sequence_axis
                ),
            )
        return out

    def pad(self, pad_length: int, fill_value: int = 0):
        assert (
            pad_length >= len(self.data)
        ), f"pad_length must be greater than or equal to the length of the sequence, got {pad_length} and {len(self.data)}"
        return SequenceWithSegments(
            data=np.pad(
                self.data,
                (0, pad_length - len(self.data)),
                mode="constant",
                constant_values=fill_value,
            ),
            mask_exists=np.pad(
                self.mask_exists, (0, pad_length - len(self.mask_exists))
            ),
            mask_ar=np.pad(self.mask_ar, (0, pad_length - len(self.mask_ar))),
            mask_loss=np.pad(self.mask_loss, (0, pad_length - len(self.mask_loss))),
            segments=self.segments,
        )

    def __getitem__(self, index):
        if isinstance(index, int):
            return Sequence(
                data=self.data[..., index],
                mask_exists=self.mask_exists[..., index],
                mask_ar=self.mask_ar[..., index],
                mask_loss=self.mask_loss[..., index],
            )
        elif isinstance(index, slice):
            slice_start = index.start or 0
            return SequenceWithSegments(
                data=self.data[..., index],
                mask_exists=self.mask_exists[..., index],
                mask_ar=self.mask_ar[..., index],
                mask_loss=self.mask_loss[..., index],
                segments={k: v - slice_start for k, v in self.segments.items()},
            )
        else:
            raise ValueError(f"Invalid index type: {type(index)}")

    def __len__(self):
        return self.data.shape[-1]
