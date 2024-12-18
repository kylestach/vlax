from dataclasses import dataclass
from typing import List, Union

import grain.python as grain
import numpy as np
from transformers import AutoTokenizer

from vlax.sequence.action_tokenizer import ActionTokenizer
from vlax.sequence.sequence import SequenceWithSegments


@dataclass
class TextSegment:
    name: str

    string: str
    autoregressive: bool
    predict_target: bool

    pad_length: int = 0


@dataclass
class SensorSegment:
    name: str
    num_tokens: int
    placeholder: int


Segment = Union[TextSegment, SensorSegment]


def assemble_sequence(language_tokenizer: AutoTokenizer, segments: List[Segment]):
    segment_markers = {}

    tokens = []
    tokens_ar = []
    tokens_loss = []
    tokens_mask = []

    for segment in segments:
        start_marker = len(tokens)
        num_tokens = None
        pad_length = None

        if isinstance(segment, TextSegment):
            new_tokens = language_tokenizer.encode(segment.string)
            autoregressive = segment.autoregressive
            predict_target = segment.predict_target
            pad_length = segment.pad_length
        elif isinstance(segment, SensorSegment):
            new_tokens = np.full(segment.num_tokens, segment.placeholder)
            autoregressive = False
            predict_target = False
            num_tokens = segment.num_tokens
        else:
            raise ValueError(f"Unknown segment type: {type(segment)}")

        tokens.extend(new_tokens)
        tokens_ar.extend([autoregressive] * len(new_tokens))
        tokens_loss.extend([predict_target] * len(new_tokens))
        tokens_mask.extend([True] * len(new_tokens))

        if pad_length is not None:
            tokens.extend([0] * (pad_length - len(new_tokens)))
            tokens_ar.extend([False] * (pad_length - len(new_tokens)))
            tokens_loss.extend([False] * (pad_length - len(new_tokens)))
            tokens_mask.extend([False] * (pad_length - len(new_tokens)))
            num_tokens = pad_length

        if num_tokens is not None:
            segment_markers[segment.name] = np.arange(
                start_marker, start_marker + num_tokens
            )

    return SequenceWithSegments(
        data=np.array(tokens),
        mask_exists=np.array(tokens_mask),
        mask_ar=np.array(tokens_ar),
        mask_loss=np.array(tokens_loss),
        segments=segment_markers,
    )


class TokenizeDataTransform(grain.MapTransform):
    def __init__(
        self,
        language_tokenizer: AutoTokenizer,
        action_tokenizer: ActionTokenizer,
        pad_sequence_length: int = 330,
        tokens_per_image: int = 256,
        action_pad_length: int = 30,
        allowed_image_keys: List[str] = None,
        allowed_proprio_keys: List[str] = None,
        allowed_language_keys: List[str] = None,
        action_keys: List[str] = None,
    ):
        """
        Create a TokenizeDataTransform.

        Args:
            language_tokenizer: The language tokenizer to use. Should have support for:
                - Action tokens (<boa>, <act#>)
                - Placeholder tokens (<image>, <proprio>) for each modality
            action_tokenizer: An ActionTokenizer.
            pad_sequence_length: The length to pad the sequence to.
            tokens_per_image: The number of tokens the encoder will use for each image.
            action_pad_length: The maximum tokenized action length.
            allowed_image_keys: The keys to use for images, or None to use all keys.
            allowed_proprio_keys: The keys to use for proprios, or None to use all keys.
            allowed_language_keys: The keys to use for language, or None to use all keys.
        """

        self.language_tokenizer = language_tokenizer
        self.action_tokenizer = action_tokenizer
        self.pad_sequence_length = pad_sequence_length
        self.tokens_per_image = tokens_per_image
        self.boa_token = self.language_tokenizer.vocab["<boa>"]
        self.eos_token = self.language_tokenizer.vocab["<eos>"]
        self.pad_token = self.language_tokenizer.vocab["<pad>"]
        self.image_placeholder = self.language_tokenizer.vocab["<image>"]
        self.proprio_placeholder = self.language_tokenizer.vocab["<proprio>"]
        self.action_pad_length = action_pad_length
        self.allowed_image_keys = allowed_image_keys
        self.allowed_proprio_keys = allowed_proprio_keys
        self.allowed_language_keys = allowed_language_keys
        self.action_keys = action_keys

    def __repr__(self):
        return f"TokenizeDataTransform(language_tokenizer={type(self.language_tokenizer)}, action_tokenizer={type(self.action_tokenizer)}, pad_length={self.pad_sequence_length})"

    def make_sequence(self, data):
        segments = []

        allowed_image_keys = (
            data["observation"]["image"].keys()
            if self.allowed_image_keys is None
            else [
                k for k in self.allowed_image_keys if k in data["observation"]["image"]
            ]
        )
        allowed_proprio_keys = (
            data["observation"]["proprio"].keys()
            if self.allowed_proprio_keys is None
            else [
                k
                for k in self.allowed_proprio_keys
                if k in data["observation"]["proprio"]
            ]
        )
        allowed_language_keys = (
            data["language"].keys()
            if self.allowed_language_keys is None
            else [k for k in self.allowed_language_keys if k in data["language"]]
        )

        for image_key in allowed_image_keys:
            segments.append(
                TextSegment(
                    name=image_key + "_begin",
                    string=image_key,
                    autoregressive=False,
                    predict_target=False,
                )
            )
            segments.append(
                SensorSegment(
                    name=image_key,
                    num_tokens=self.tokens_per_image,
                    placeholder=self.image_placeholder,
                )
            )

        for proprio_key in allowed_proprio_keys:
            segments.append(
                TextSegment(
                    name=proprio_key + "_begin",
                    string=proprio_key,
                    autoregressive=False,
                    predict_target=False,
                )
            )
            segments.append(
                SensorSegment(
                    name=proprio_key,
                    num_tokens=1,
                    placeholder=self.proprio_placeholder,
                )
            )

        segments.append(
            TextSegment(
                name="bos", string="<bos>", autoregressive=False, predict_target=False
            )
        )

        for language_key in allowed_language_keys:
            # Language prompt instruction
            language_instruction = data["language"][language_key]
            if isinstance(language_instruction, np.ndarray):
                language_instruction = language_instruction.item()
            if isinstance(language_instruction, bytes):
                language_instruction = language_instruction.decode("utf-8")

        segments.append(
            TextSegment(
                name="language_instruction",
                string=language_instruction,
                autoregressive=False,
                predict_target=False,
            )
        )

        # Action
        if self.action_keys is None:
            action = data["action"]
        else:
            action = np.concatenate(
                [data["action"][key] for key in self.action_keys], axis=-1
            )

        action_component = self.action_tokenizer.tokenize(action)
        action_string = (
            "<boa>" + "".join(f"<act{i}>" for i in action_component) + "<eos>"
        )
        segments.append(
            TextSegment(
                name="action",
                string=action_string,
                autoregressive=True,
                predict_target=True,
                pad_length=self.action_pad_length,
            )
        )

        # Assemble the sequence
        return assemble_sequence(self.language_tokenizer, segments)

    def map(self, data):
        # Pad the tokens to the sequence length
        sequence = self.make_sequence(data)
        action_start = sequence.segments["action"][0]
        sequence = {
            "train_tokens": sequence.pad(
                self.pad_sequence_length, fill_value=self.pad_token
            ),
            "prompt_tokens": sequence[:action_start].pad(
                self.pad_sequence_length, fill_value=self.pad_token
            ),
            "gen_tokens": sequence[action_start:].pad(
                self.pad_sequence_length, fill_value=self.pad_token
            ),
        }
        return data | sequence
