from typing import List, Tuple

from grain.python import Batch, MapDataset, MapTransform
from grain_oxe.core.structure import DATASET_STRUCTURES
from grain_oxe.datasets.create import create_standardized_dataset
from grain_oxe.datasets.normalization import NormalizationType, Normalizer
from grain_oxe.frame_transforms import DecodeImageTransform
from grain_oxe.mask_utils import ExtractMaskTransform

from vlax.sequence.action_tokenizer import ActionTokenizer
from vlax.sequence.sequence_builder import TokenizeDataTransform


def make_single_dataset(
    language_tokenizer: ActionTokenizer,
    action_tokenizer: ActionTokenizer,
    tokenize_transform_kwargs: dict,
    dataset_name: str,
    data_dir: str,
    data_stats_dir: str,
    dataset_structure: str,
    dataset_structure_kwargs: dict,
    proprio_normalization_type: NormalizationType,
    action_normalization_type: NormalizationType,
    batch_size: int,
    image_size: Tuple[int, int],
    split: str,
) -> Tuple[MapDataset, List[MapTransform]]:
    normalizer = Normalizer.load(
        data_stats_dir,
        {
            "observation": {"proprio": proprio_normalization_type},
            "action": {"proprio": action_normalization_type},
        },
    )

    dataset = create_standardized_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        dataset_structure=DATASET_STRUCTURES[dataset_structure](
            **dataset_structure_kwargs
        ),
        split=split,
        seed=0,
        normalizer=normalizer,
    )

    transforms = [
        DecodeImageTransform(
            **{
                "*image/*": dict(resize_size=image_size),
            }
        ),
        TokenizeDataTransform(
            language_tokenizer, action_tokenizer, **tokenize_transform_kwargs
        ),
        ExtractMaskTransform(),
        Batch(batch_size),
    ]

    return dataset, transforms
