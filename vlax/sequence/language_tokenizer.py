from typing import List

from transformers import AutoTokenizer

from vlax.sequence.action_tokenizer import ActionTokenizer


def build_language_tokenizer(
    base_tokenizer: AutoTokenizer,
    action_tokenizer: ActionTokenizer,
    extra_modalities: List[str],
) -> AutoTokenizer:
    language_tokenizer = base_tokenizer

    # Don't automatically add <bos>, we will do this manually
    language_tokenizer.add_bos_token = False

    language_tokenizer.add_tokens(
        [f"<act{i}>" for i in range(action_tokenizer.vocab_size)]
        + ["<boa>"]
        + [f"<{modality}>" for modality in extra_modalities]
    )

    return language_tokenizer
