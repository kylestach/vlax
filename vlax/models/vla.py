import importlib
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

import chex
import jax.numpy as jnp
from flax.core import FrozenDict

from vlax.models.big_vision import PaliGemmaModel
from vlax.models.big_vision.paligemma import make_attn_mask
from vlax.sequence.sequence import Sequence, SequenceWithSegments


class Model(PaliGemmaModel):
    _extra_encoder_models: Optional[FrozenDict[str, str]] = None
    _extra_encoder_model_kwargs: Optional[FrozenDict[str, FrozenDict]] = None

    modality_mapping: Optional[FrozenDict[str, str]] = None

    def setup(self):
        super().setup()

        encoders = {}
        if self._extra_encoder_models is not None:
            for encoder, model in self._extra_encoder_models.items():
                *model_module, model_class = model.split(".")
                model_module = ".".join(model_module)
                model_class = getattr(
                    importlib.import_module(model_module), model_class
                )
                encoders[encoder] = model_class(
                    **self._extra_encoder_model_kwargs.get(encoder, FrozenDict()),
                    name=encoder,
                )

        def _embed_img(img, mask=None):
            if mask is None:
                mask = jnp.ones(img.shape[:-3], dtype=jnp.bool_)
            mask = jnp.all(mask, axis=tuple(range(img.ndim - 3, mask.ndim)))

            batch_shape = img.shape[:-3]
            img_ = img.reshape(-1, *img.shape[-3:])
            embeddings, _ = self._img_model(img_)
            embeddings = embeddings.reshape(batch_shape[0], -1, embeddings.shape[-1])

            mask = jnp.repeat(mask[..., None], embeddings.shape[-2], axis=-1).reshape(
                *embeddings.shape[:-2], -1
            )

            return embeddings, mask

        encoders["img"] = _embed_img

        self.encoders = encoders

    def embed_inputs(
        self, inputs: Dict[str, Any], masks: Dict[str, Any]
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        embeddings = {}
        embed_masks = {}

        for modality, values in inputs.items():
            mask_values = masks.get(modality, None)
            for key, value in values.items():
                if modality not in self.modality_mapping:
                    continue

                mask = mask_values.get(key, None)
                encoder = self.encoders[self.modality_mapping[modality]]
                embeddings[key], embed_masks[key] = encoder(value, mask)

                chex.assert_shape(embeddings[key], embed_masks[key].shape + (None,))

        return embeddings, embed_masks

    def assemble_sequence(
        self,
        sequence: SequenceWithSegments,
        embeddings: Dict[str, jnp.ndarray],
        embed_masks: Dict[str, jnp.ndarray],
    ) -> Sequence:
        return sequence.fill_jax(segment_values=embeddings, segment_masks=embed_masks)

    def make_inputs(
        self,
        sequence: SequenceWithSegments,
        data: Dict[str, jnp.ndarray],
        data_mask: Dict[str, jnp.ndarray],
    ):
        # Embed the image and text.
        embeds, embed_masks = self.embed_inputs(data, data_mask)
        out = {f"embeds/{key}": embeds[key] for key in embeds}

        sequence_embeds = replace(sequence, data=self._llm.embed_tokens(sequence.data))
        sequence_embeds = self.assemble_sequence(sequence_embeds, embeds, embed_masks)
        return sequence_embeds, out

    def __call__(self, sequence, data, data_mask, train=False, pre_logits=False):
        """Concats image/text and returns text logits.

        Args:
          image: float32[B, H, W, 3] image that can be passed to the `img` model.
          text: int32[B, T] token sequence that can be embedded by the `txt` model.
          mask_ar: int32[B, T] mask that's 1 where `text` should be attended to
            causally, and 0 where it can be attended to with full self-attention.
          train: bool whether we're in train or test mode (dropout etc).

        Returns:
          float32[B, T, V] logits for the `text` input, and an out-dict of named
          intermediates.
        """
        sequence_embeds, out = self.make_inputs(sequence, data, data_mask)

        # Call transformer on the embedded token sequence.
        attn_mask = out["attn_mask"] = make_attn_mask(
            sequence_embeds.mask_exists, sequence_embeds.mask_ar
        )
        all_pre_logits, out_llm = self._llm(
            sequence_embeds.data, mask=attn_mask, train=train
        )
        pre_logits_sequence = replace(sequence, data=all_pre_logits)

        for k, v in out_llm.items():
            out[f"llm/{k}"] = v

        # Extract the logits for each segment
        pre_logits_segments = pre_logits_sequence.extract_segments()
        for k, v in pre_logits_segments.items():
            out[f"pre_logits/{k}"] = v.data

        if pre_logits:
            result_segments = pre_logits_segments
        else:
            logits = {
                k: self._llm.compute_logits(v.data, train=train)
                for k, v in pre_logits_segments.items()
            }
            for k, v in logits.items():
                out[f"logits/{k}"] = v
            result_segments = logits

        return all_pre_logits, result_segments, out

    def compute_logits(self, pre_logits, train=False):
        return self._llm.compute_logits(pre_logits, train=train)

    def init_helper(self, example_batch):
        used_encoders = set()
        for modality, encoder_name in self.modality_mapping.items():
            encoder = self.encoders[encoder_name]
            _ = encoder(example_batch[modality], None)
            used_encoders.add(encoder_name)

        assert (
            set(self.encoders.keys()) == used_encoders
        ), f"Missing encoders: {set(self.encoders.keys()) - used_encoders} during init"

        # Pass through the LLM
        seq = jnp.arange(10)[None]
        mask = jnp.ones_like(seq)
        attn_mask = make_attn_mask(mask, mask)
        _ = self._llm(self._llm.embed_tokens(seq), mask=attn_mask, train=False)
